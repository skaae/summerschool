__author__ = 'casperkaae'
import lasagne

lasagne.layers.EmbeddingLayer

from data_generator import get_batch, print_valid_characters
batch_size = 10
inputs, input_masks, targets, target_masks, text_inputs, text_targets = \
    get_batch(batch_size=batch_size,max_digits=2,min_digits=1)



print print_valid_characters()
print "Stop character = #"

for i in range(batch_size):
    print "\nSAMPLE",i
    print "TEXT INPUTS:\t\t", text_inputs[i]
    print "TEXT TARGETS:\t\t", text_targets[i]
    print "ENCODED INPUTS:\t\t", inputs[i]
    print "MASK INPUTS:\t\t", input_masks[i]
    print "ENCODED TARGETS:\t", targets[i]
    print "MASKNPUTS:\t\t\t", target_masks[i]