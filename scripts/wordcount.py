import re
files = ['sec_intro.tex', 'sec_theory.tex', 'sec_method.tex', 'sec_results.tex',
         'sec_evaluation.tex', 'sec_conclusion.tex']
tot = 0
for f in files:
    s = open(f, encoding='utf-8').read()
    s = re.sub(r'%.*', '', s)
    s = re.sub(r'\\(begin|end)\{[^}]*\}', ' ', s)
    s = re.sub(r'\\(parencite|textcite|cite)\{[^}]*\}', ' CITE ', s)   # 1 word each
    s = re.sub(r'\\[A-Z][A-Za-z]+', ' NUM ', s)                        # value macros -> 1 word
    s = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?', ' ', s)     # other commands
    s = re.sub(r'[\\{}$&~_^]', ' ', s)
    w = len([x for x in s.split() if re.search(r'[A-Za-z0-9]', x)])
    print(f'{f}: {w}')
    tot += w
print('BODY TOTAL (cites + value-macros each = 1 word):', tot)
