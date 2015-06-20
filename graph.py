from collections import deque


class node():
    def __init__(self, input, layer):
        self.layer = layer
        self.input = input

    def fprop(self, x):
        return self.layer.fprop(x)

    def bprop(self, delta_in):
        return self.layer.bprop(delta_in)

    def update_params(self, lr):
        self.layer.update_params(lr)


def fprop(topnode):
    nodes = []
    node = topnode
    while node.input is not None:
        node = node.input
        nodes += [node]






def get_all_layers(layer, treat_as_input=None):
    try:
        queue = deque(layer)
    except TypeError:
        queue = deque([layer])
    seen = set()
    done = set()
    result = []

    # If treat_as_input is given, we pretend we've already collected all their
    # incoming layers.
    if treat_as_input is not None:
        seen.update(treat_as_input)

    while queue:
        # Peek at the leftmost node in the queue.
        layer = queue[0]
        if layer is None:
            # Some node had an input_layer set to `None`. Just ignore it.
            queue.popleft()
        elif layer not in seen:
            # We haven't seen this node yet: Mark it and queue all incomings
            # to be processed first. If there are no incomings, the node will
            # be appended to the result list in the next iteration.
            seen.add(layer)
            if hasattr(layer, 'input_layers'):
                queue.extendleft(reversed(layer.input_layers))
            elif hasattr(layer, 'input_layer'):
                queue.appendleft(layer.input_layer)
        else:
            # We've been here before: Either we've finished all its incomings,
            # or we've detected a cycle. In both cases, we remove the layer
            # from the queue and append it to the result list.
            queue.popleft()
            if layer not in done:
                result.append(layer)
                done.add(layer)

    return result


