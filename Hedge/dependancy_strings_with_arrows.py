def string_with_arrows(text, posStart, posEnd):
    result = ''

    # Calculate indices
    indexStart = max(text.rfind('\n', 0, posStart.index), 0)
    indexEnd = text.find('\n', indexStart + 1)
    if indexEnd < 0: indexEnd = len(text)
    
    # Generate each line
    lineCount = posEnd.line - posStart.line + 1
    for i in range(lineCount):
        # Calculate line columns
        line = text[indexStart:indexEnd]
        columnStart = posStart.column if i == 0 else 0
        columnEnd = posEnd.column if i == lineCount - 1 else len(line) - 1

        # Append to result
        result += line + '\n'
        result += ' ' * columnStart + '^' * (columnEnd - columnStart)

        # Re-calculate indices
        indexStart = indexEnd
        indexEnd = text.find('\n', indexStart + 1)
        if indexEnd < 0: indexEnd = len(text)

    return result.replace('\t', '')
