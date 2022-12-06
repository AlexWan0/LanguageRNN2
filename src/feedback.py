from nltk.metrics.distance import edit_distance
import feedback

def closest_target(cleaned_pred, target_utters):
    utter_dists = [(-edit_distance(t, cleaned_pred), t) for t in target_utters]
    utter_closest = max(utter_dists, key=lambda x:x[0])[1] + ' !'
    
    return utter_closest

def feedback_target(cleaned_pred, target_utters):
    utter_feedback = feedback.get_feedback_sentence(target_utters, cleaned_pred) + ' !'
    
    return utter_feedback
