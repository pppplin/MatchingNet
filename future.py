from tensorflow.python.util import nest
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn import bidirectional_rnn
from tensorflow.python.ops import array_ops

def _like_rnncell(cell):
  conditions = [hasattr(cell, "output_size"), hasattr(cell, "state_size"), hasattr(cell, "zero_state"), callable(cell)]
  return all(conditions)

def _reverse_seq(input_seq, lengths):
  """Reverse a list of Tensors up to specified lengths.

  Args:
    input_seq: Sequence of seq_len tensors of dimension (batch_size, n_features)
               or nested tuples of tensors.
    lengths:   A `Tensor` of dimension batch_size, containing lengths for each
               sequence in the batch. If "None" is specified, simply reverses
               the list.

  Returns:
    time-reversed sequence
  """
  if lengths is None:
    return list(reversed(input_seq))

  flat_input_seq = tuple(nest.flatten(input_) for input_ in input_seq)

  flat_results = [[] for _ in range(len(input_seq))]
  for sequence in zip(*flat_input_seq):
    input_shape = tensor_shape.unknown_shape(
        ndims=sequence[0].get_shape().ndims)
    for input_ in sequence:
      input_shape.merge_with(input_.get_shape())
      input_.set_shape(input_shape)

    # Join into (time, batch_size, depth)
    s_joined = array_ops.stack(sequence)

    # TODO(schuster, ebrevdo): Remove cast when reverse_sequence takes int32
    if lengths is not None:
      lengths = math_ops.to_int64(lengths)

    # Reverse along dimension 0
    s_reversed = array_ops.reverse_sequence(s_joined, lengths, 0, 1)
    # Split again into list
    result = array_ops.unstack(s_reversed)
    for r, flat_result in zip(result, flat_results):
      r.set_shape(input_shape)
      flat_result.append(r)

  results = [nest.pack_sequence_as(structure=input_, flat_sequence=flat_result)
             for input_, flat_result in zip(input_seq, flat_results)]
  return results


def static_rnn(cell,
               inputs,
               initial_state=None,
               dtype=None,
               sequence_length=None,
               scope=None):
  """Creates a recurrent neural network specified by RNNCell `cell`.

  The simplest form of RNN network generated is:

  ```python
    state = cell.zero_state(...)
    outputs = []
    for input_ in inputs:
      output, state = cell(input_, state)
      outputs.append(output)
    return (outputs, state)
  ```
  However, a few other options are available:

  An initial state can be provided.
  If the sequence_length vector is provided, dynamic calculation is performed.
  This method of calculation does not compute the RNN steps past the maximum
  sequence length of the minibatch (thus saving computational time),
  and properly propagates the state at an example's sequence length
  to the final state output.

  The dynamic calculation performed is, at time `t` for batch row `b`,

  ```python
    (output, state)(b, t) =
      (t >= sequence_length(b))
        ? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
        : cell(input(b, t), state(b, t - 1))
  ```

  Args:
    cell: An instance of RNNCell.
    inputs: A length T list of inputs, each a `Tensor` of shape
      `[batch_size, input_size]`, or a nested tuple of such elements.
    initial_state: (optional) An initial state for the RNN.
      If `cell.state_size` is an integer, this must be
      a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
      If `cell.state_size` is a tuple, this should be a tuple of
      tensors having shapes `[batch_size, s] for s in cell.state_size`.
    dtype: (optional) The data type for the initial state and expected output.
      Required if initial_state is not provided or RNN state has a heterogeneous
      dtype.
    sequence_length: Specifies the length of each sequence in inputs.
      An int32 or int64 vector (tensor) size `[batch_size]`, values in `[0, T)`.
    scope: VariableScope for the created subgraph; defaults to "rnn".

  Returns:
    A pair (outputs, state) where:

    - outputs is a length T list of outputs (one for each input), or a nested
      tuple of such elements.
    - state is the final state

  Raises:
    TypeError: If `cell` is not an instance of RNNCell.
    ValueError: If `inputs` is `None` or an empty list, or if the input depth
      (column size) cannot be inferred from inputs via shape inference.
  """

  if not _like_rnncell(cell):
    raise TypeError("cell must be an instance of RNNCell")
  if not nest.is_sequence(inputs):
    raise TypeError("inputs must be a sequence")
  if not inputs:
    raise ValueError("inputs must not be empty")

  outputs = []
  # Create a new scope in which the caching device is either
  # determined by the parent scope, or is set to place the cached
  # Variable using the same placement as for the rest of the RNN.
  with vs.variable_scope(scope or "rnn") as varscope:
    if varscope.caching_device is None:
      varscope.set_caching_device(lambda op: op.device)

    # Obtain the first sequence of the input
    first_input = inputs
    while nest.is_sequence(first_input):
      first_input = first_input[0]

    # Temporarily avoid EmbeddingWrapper and seq2seq badness
    # TODO(lukaszkaiser): remove EmbeddingWrapper
    if first_input.get_shape().ndims != 1:

      input_shape = first_input.get_shape().with_rank_at_least(2)
      fixed_batch_size = input_shape[0]

      flat_inputs = nest.flatten(inputs)
      for flat_input in flat_inputs:
        input_shape = flat_input.get_shape().with_rank_at_least(2)
        batch_size, input_size = input_shape[0], input_shape[1:]
        fixed_batch_size.merge_with(batch_size)
        for i, size in enumerate(input_size):
          if size.value is None:
            raise ValueError(
                "Input size (dimension %d of inputs) must be accessible via "
                "shape inference, but saw value None." % i)
    else:
      fixed_batch_size = first_input.get_shape().with_rank_at_least(1)[0]

    if fixed_batch_size.value:
      batch_size = fixed_batch_size.value
    else:
      batch_size = array_ops.shape(first_input)[0]
    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If no initial_state is provided, "
                         "dtype must be specified")
      state = cell.zero_state(batch_size, dtype)

    if sequence_length is not None:  # Prepare variables
      sequence_length = ops.convert_to_tensor(
          sequence_length, name="sequence_length")
      if sequence_length.get_shape().ndims not in (None, 1):
        raise ValueError(
            "sequence_length must be a vector of length batch_size")

      def _create_zero_output(output_size):
        # convert int to TensorShape if necessary
        size = _concat(batch_size, output_size)
        output = array_ops.zeros(
            array_ops.stack(size), _infer_state_dtype(dtype, state))
        shape = _concat(fixed_batch_size.value, output_size, static=True)
        output.set_shape(tensor_shape.TensorShape(shape))
        return output

      output_size = cell.output_size
      flat_output_size = nest.flatten(output_size)
      flat_zero_output = tuple(
          _create_zero_output(size) for size in flat_output_size)
      zero_output = nest.pack_sequence_as(
          structure=output_size, flat_sequence=flat_zero_output)

      sequence_length = math_ops.to_int32(sequence_length)
      min_sequence_length = math_ops.reduce_min(sequence_length)
      max_sequence_length = math_ops.reduce_max(sequence_length)

    for time, input_ in enumerate(inputs):
      if time > 0:
        varscope.reuse_variables()
      # pylint: disable=cell-var-from-loop
      call_cell = lambda: cell(input_, state)
      # pylint: enable=cell-var-from-loop
      if sequence_length is not None:
        (output, state) = _rnn_step(
            time=time,
            sequence_length=sequence_length,
            min_sequence_length=min_sequence_length,
            max_sequence_length=max_sequence_length,
            zero_output=zero_output,
            state=state,
            call_cell=call_cell,
            state_size=cell.state_size)
      else:
        (output, state) = call_cell()

      outputs.append(output)

    return (outputs, state)

def static_bidirectional_rnn(cell_fw,
                             cell_bw,
                             inputs,
                             initial_state_fw=None,
                             initial_state_bw=None,
                             dtype=None,
                             sequence_length=None,
                             scope=None):
  """Creates a bidirectional recurrent neural network.
  Similar to the unidirectional case above (rnn) but takes input and builds
  independent forward and backward RNNs with the final forward and backward
  outputs depth-concatenated, such that the output will have the format
  [time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
  forward and backward cell must match. The initial state for both directions
  is zero by default (but can be set optionally) and no intermediate states are
  ever returned -- the network is fully unrolled for the given (passed in)
  length(s) of the sequence(s) or completely unrolled if length(s) is not given.
  Args:
    cell_fw: An instance of RNNCell, to be used for forward direction.
    cell_bw: An instance of RNNCell, to be used for backward direction.
    inputs: A length T list of inputs, each a tensor of shape
      [batch_size, input_size], or a nested tuple of such elements.
    initial_state_fw: (optional) An initial state for the forward RNN.
      This must be a tensor of appropriate type and shape
      `[batch_size, cell_fw.state_size]`.
      If `cell_fw.state_size` is a tuple, this should be a tuple of
      tensors having shapes `[batch_size, s] for s in cell_fw.state_size`.
    initial_state_bw: (optional) Same as for `initial_state_fw`, but using
      the corresponding properties of `cell_bw`.
    dtype: (optional) The data type for the initial state.  Required if
      either of the initial states are not provided.
    sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
      containing the actual lengths for each of the sequences.
    scope: VariableScope for the created subgraph; defaults to
      "bidirectional_rnn"
  Returns:
    A tuple (outputs, output_state_fw, output_state_bw) where:
      outputs is a length `T` list of outputs (one for each input), which
        are depth-concatenated forward and backward outputs.
      output_state_fw is the final state of the forward rnn.
      output_state_bw is the final state of the backward rnn.
  Raises:
    TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
    ValueError: If inputs is None or an empty list.
  """

  if not _like_rnncell(cell_fw):
    raise TypeError("cell_fw must be an instance of RNNCell")
  if not _like_rnncell(cell_bw):
    raise TypeError("cell_bw must be an instance of RNNCell")

  if not nest.is_sequence(inputs):
    raise TypeError("inputs must be a sequence")
  if not inputs:
    raise ValueError("inputs must not be empty")

  with vs.variable_scope(scope or "bidirectional_rnn"):
    # Forward direction
    with vs.variable_scope("fw") as fw_scope:
      output_fw, output_state_fw = static_rnn(
          cell_fw,
          inputs,
          initial_state_fw,
          dtype,
          sequence_length,
          scope=fw_scope)

    # Backward direction
    with vs.variable_scope("bw") as bw_scope:
      reversed_inputs = _reverse_seq(inputs, sequence_length)
      tmp, output_state_bw = static_rnn(
          cell_bw,
          reversed_inputs,
          initial_state_bw,
          dtype,
          sequence_length,
          scope=bw_scope)

  output_bw = _reverse_seq(tmp, sequence_length)
  # Concat each of the forward/backward outputs
  flat_output_fw = nest.flatten(output_fw)
  flat_output_bw = nest.flatten(output_bw)

  flat_outputs = tuple(
      # array_ops.concat([fw, bw], 1)
      # int32/float32
      array_ops.concat(1, [fw, bw])
      for fw, bw in zip(flat_output_fw, flat_output_bw))

  outputs = nest.pack_sequence_as(
      structure=output_fw, flat_sequence=flat_outputs)

  return (outputs, output_state_fw, output_state_bw)


def stack_bidirectional_rnn(cells_fw,
                            cells_bw,
                            inputs,
                            initial_states_fw=None,
                            initial_states_bw=None,
                            dtype=None,
                            sequence_length=None,
                            scope=None):

# in 1.* tensorflow but not in 0.11. bidirectionalLSTM with communication. 
    if not cells_fw:
        raise ValueError("Must specify at least one fw cell for BidirectionalRNN.")
    if not cells_bw:
        raise ValueError("Must specify at least one bw cell for BidirectionalRNN.")
    if not isinstance(cells_fw, list):
        raise ValueError("cells_fw must be a list of RNNCells (one per layer).")
    if not isinstance(cells_bw, list):
        raise ValueError("cells_bw must be a list of RNNCells (one per layer).")
    if len(cells_fw) != len(cells_bw):
        raise ValueError("Forward and Backward cells must have the same depth.")
    if initial_states_fw is not None and (not isinstance(cells_fw, list) or
                                            len(cells_fw) != len(cells_fw)):
        raise ValueError(
            "initial_states_fw must be a list of state tensors (one per layer).")
    if initial_states_bw is not None and (not isinstance(cells_bw, list) or
                                            len(cells_bw) != len(cells_bw)):
        raise ValueError(
            "initial_states_bw must be a list of state tensors (one per layer).")
    states_fw = []
    states_bw = []
    prev_layer = inputs

    with vs.variable_scope(scope or "stack_bidirectional_rnn"):
        for i, (cell_fw, cell_bw) in enumerate(zip(cells_fw, cells_bw)):
            initial_state_fw = None
            initial_state_bw = None
            if initial_states_fw:
                initial_state_fw = initial_states_fw[i]
            if initial_states_bw:
                initial_state_bw = initial_states_bw[i]

            with vs.variable_scope("cell_%d" % i) as cell_scope:
                # bidirectional_rnn
                # prev_layer, state_fw, state_bw = rnn.static_bidirectional_rnn(
                prev_layer, state_fw, state_bw = rnn.bidirectional_rnn(
                    cell_fw,
                    cell_bw,
                    prev_layer,
                    initial_state_fw=initial_state_fw,
                    initial_state_bw=initial_state_bw,
                    sequence_length=sequence_length,
                    dtype=dtype,
                    scope=cell_scope)
            states_fw.append(state_fw)
            states_bw.append(state_bw)

    return prev_layer, tuple(states_fw), tuple(states_bw)
