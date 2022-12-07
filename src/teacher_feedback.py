import re

# shape + attributes - capturing group 1
# (the ((small|medium|big) )?((blue|red|yellow|green|purple|orange) )?(ellipse|circle|star|square|hexagon|triangle))

# position - capturing group 8
#  (above|below|right of|left of) 

# shape + attributes 2 - capturing group 9
# (the ((small|medium|big) )?((blue|red|yellow|green|purple|orange) )?(ellipse|circle|star|square|hexagon|triangle))

pattern = r'(the ((small|medium|big) )?((blue|red|yellow|green|purple|orange) )?(ellipse|circle|star|square|hexagon|triangle))( (above|below|to the right of|to the left of) (the ((small|medium|big) )?((blue|red|yellow|green|purple|orange) )?(ellipse|circle|star|square|hexagon|triangle)))?'

def get_shapes(utter):
    x = re.search(pattern, utter)
    
    if not x:
        return None
    
    shape1 = [
        x.group(3),
        x.group(5),
        x.group(6)
    ]
    
    shape2 = [
        x.group(11),
        x.group(13),
        x.group(14)
    ]
    
    pos = x.group(8)
    
    return shape1, pos, shape2
    
def shape_dist(shape1, shape2):
    dist = 0
    for attr1, attr2 in zip(shape1, shape2):
        if attr1 != attr2:
            dist += 1
    return dist

flip_position = {
    'above': 'below',
    'below': 'above',
    'to the right of': 'to the left of',
    'to the left of': 'to the right of'
}

def get_closest_shape(sentence, shape, sentence_pos):
    shape1_dist = shape_dist(sentence[0], shape)
    shape2_dist = shape_dist(sentence[2], shape)

    # select first shape
    if shape1_dist < shape2_dist:
        if sentence_pos == 0: # if shapes are both in the first slot of the sentence
            pos = sentence[1]
        else:
            pos = flip_position[sentence[1]]

        return sentence[0], shape1_dist, pos
    
    if sentence_pos == 1: # if shapes are both in the second slot of the sentence
        pos = sentence[1]
    else:
        pos = flip_position[sentence[1]]

    # select second shape
    return sentence[2], shape2_dist, pos

def get_shape_str(shape):
    no_none = [s for s in shape if s != None]
    return 'the ' + ' '.join(no_none)

def reconstruct_str(sentence):
    shape1_str = get_shape_str(sentence[0])

    if sentence[1] is None:
        return shape1_str

    shape2_str = get_shape_str(sentence[2])

    return shape1_str + ' ' + sentence[1] + ' ' + shape2_str

def get_feedback_sentence(target_utters, predicted):
    target_max = max(target_utters, key=lambda u:len(u)) 
    
    pred_sentence = get_shapes(predicted)
    target_sentence = get_shapes(target_max)
    
    if pred_sentence is None:
        return target_max
    
    feedback_sentence = [None, None, None]
    
    feedback_sentence[0], shape1_dist, shape1_pos = get_closest_shape(target_sentence, pred_sentence[0], 0)
    
    feedback_sentence[2], shape2_dist, shape2_pos = get_closest_shape(target_sentence, pred_sentence[2], 1)
    
    '''
    # update shape with largest distance to target
    if shape1_dist > shape2_dist: # update shape 1 only
        feedback_sentence[1] = shape1_pos
        feedback_sentence[2] = pred_sentence[2]
    else: # update shape 2 only
        feedback_sentence[1] = shape2_pos
        feedback_sentence[0] = pred_sentence[0]
    '''
    
    feedback_sentence[1] = pred_sentence[1]

    return reconstruct_str(feedback_sentence)
