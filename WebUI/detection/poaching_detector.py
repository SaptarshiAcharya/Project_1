def detect_poaching_activity(detected_animals):
    """
    Rule-based placeholder:
    If no animals for long time OR human detected → possible poaching activity.
    """
    animals = [a for a, _ in detected_animals]
    if "person" in animals:
        return True, "Human detected in restricted zone!"
    if len(animals) == 0:
        return False, "No animals detected currently."
    return False, "Normal activity."
