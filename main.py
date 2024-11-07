import re
n = int(input())
pattern1 = r'^[456]\d{15}$|^[456]\d{3}-\d{4}-\d{4}-\d{4}$'
pattern2 = r'(\d)\1{3,}|(\d)\2{1}-(\d)\3{1}|-(\d)\4{3,}-'
for i in range(n):
    s = input()
    if (re.search(pattern1, s)):
        if (re.search(pattern2, s)):
            print('Invalid')
        else:
            print('Valid')