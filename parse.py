f = open('success_rate.txt')
lines = f.readlines()
ll = []
for l in lines:
    ll.append(l.split(','))

lines = []
for l in ll:
    for i in l:
        lines.append(i.split(':'))

results = []
res = []
for l in lines:
    if 'n =' in l[0]:
        res.append(int(l[0].split()[-1]))
    if 'k =' in l[0]:
        res.append(int(l[0].split()[-1]))
    try:
        if 'success =' in l[1]:
            l = l[1].split('=')[-1]
            l = l.split('/')
            res.append( float(l[0]) / float(l[1]) )
    except IndexError:
        pass

    if '\n' in l[0] and res != []:
        results.append(tuple(res))
        res = []

results.append((9,9,0))

#fo = open('success.tex','w')

text = '''\\documentclass{standalone}
\\usepackage{tikz}
\\usepackage{trdean}

\\begin{document}
\\begin{tikzpicture}
\\begin{axis}[
    grid=both,
    xlabel={$k$},
    ylabel={Success Probability}
]\n'''

current_n = 2
data = []
for t in results:
    if t[0] != current_n:
        current_n += 1
        text += '\\addplot coordinates\n{\n'
        for d in data:
            text += '(%f, %f)\n' % d
        text += '};\n\n'
        data = [(t[1],t[2])]
        continue

    data.append( (t[1],t[2]) )

text += '''\\end{axis}
\\end{tikzpicture}
\\end{document}'''
