from nltk.metrics.distance import edit_distance
import teacher_feedback
import random

# need double forward pass
def closest_target(cleaned_pred, target_utters):
    utter_dists = [(-edit_distance(t, cleaned_pred), t) for t in target_utters]
    utter_closest = max(utter_dists, key=lambda x:x[0])[1] + ' !'
    
    return utter_closest

def feedback_target(cleaned_pred, target_utters):
    utter_feedback = teacher_feedback.get_feedback_sentence(target_utters, cleaned_pred) + ' !'
    
    return utter_feedback

# no double forward pass
def max_target(target_utters):
    return max(target_utters, key=lambda x:len(x.split(' '))) + ' !'

def random_target(target_utters):
    return random.choice(target_utters) + ' !'

feedback_double_forward  = {
    'feedback': feedback_target,
    'closest': closest_target
}

feedback_single_forward  = {
    'random': random_target,
    'max': max_target
}
