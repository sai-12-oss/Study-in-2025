def edit_distance(str1, str2):
    """
    Given two strings str1 and str2, this function returns the minimum number of insertions and deletions to convert str1 to str2,
    along with the positions of insertions and deletions.
    """
    m = len(str1)
    n = len(str2)
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                dp[i][j] = i + j
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1])

    i, j = m, n
    insertions = []
    deletions = []
    while i > 0 and j > 0:
        if str1[i-1] == str2[j-1]:
            i -= 1
            j -= 1
        elif dp[i-1][j] < dp[i][j-1]:
            deletions.append(i-1)
            i -= 1
        else:
            insertions.append(j-1)
            j -= 1

    while i > 0:
        deletions.append(i-1)
        i -= 1

    while j > 0:
        insertions.append(j-1)
        j -= 1

    return insertions, deletions
