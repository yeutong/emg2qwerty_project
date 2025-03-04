from emg2qwerty.charset import charset

def calculate_cer(predictions, targets):
    """
    Calculate Character Error Rate
    """
    total_edits = 0
    total_length = 0

    for pred, target in zip(predictions, targets):
        # Remove blank tokens and duplicates
        filtered_pred = []
        prev = None
        for p in pred:
            if p != charset().null_class and p != prev:
                filtered_pred.append(p)
            prev = p

        # Calculate edit distance
        edit_distance = levenshtein_distance(filtered_pred, target)

        total_edits += edit_distance
        total_length += len(target)

    # Return CER
    return total_edits / max(1, total_length)

def levenshtein_distance(s1, s2):
    """
    Calculate Levenshtein distance between two sequences
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

