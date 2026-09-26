def calculate_fusion_score(eeg_score, heart_rate, motion_level):
    """
    Experimental multi-sensor fusion score.

    This is an academic prototype only and is not clinically validated.
    """

    eeg_component = eeg_score * 0.70

    if 60 <= heart_rate <= 100:
        heart_component = 20
    else:
        heart_component = 10

    motion_component = (1 - motion_level) * 10

    score = eeg_component + heart_component + motion_component

    return round(max(0, min(score, 100)), 1)


def fusion_state(score):
    if score >= 70:
        return "Focused"
    elif score >= 40:
        return "Moderate"
    else:
        return "Distracted"


if __name__ == "__main__":
    score = calculate_fusion_score(
        eeg_score=65,
        heart_rate=78,
        motion_level=0.25
    )

    print("Fusion Score:", score)
    print("State:", fusion_state(score))
