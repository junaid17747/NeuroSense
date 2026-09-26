def attention_score(attention_index):
    """
    Convert experimental attention index into a simple 0-100 score.

    Prototype mapping only.
    This is NOT a clinically validated score.
    """

    score = attention_index * 100

    if score > 100:
        score = 100

    if score < 0:
        score = 0

    return round(score, 1)


def attention_state(score):
    """
    Classify prototype attention level.
    """

    if score >= 70:
        return "Focused"

    elif score >= 40:
        return "Moderate"

    else:
        return "Distracted"


if __name__ == "__main__":

    test_index = 0.65

    score = attention_score(test_index)
    state = attention_state(score)

    print("Attention Score:", score)
    print("Attention State:", state)