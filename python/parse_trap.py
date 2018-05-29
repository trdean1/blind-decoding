import re
import numpy as np

def parse_matrix(text):
    #Parse out to individual rows
    rows = re.findall('\[+(.*?)\]',text)
    float_list = []
    for row in rows:
        #trim leading and trailing whitespace
        while row[0] == ' ':
            row = row[1:]
        while row[-1] == ' ':
            row = row[:-1]
        row = row.replace(',', ' ')
        try: 
            float_list.append([float(f) for f in re.split(' +',row)])
        except ValueError:
            print 'Warning skipping entry'
            #print rows
            return ''

    return np.matrix(float_list)

class trap_case:
    def __init__(self, text):
        self.U = ''
        self.Ainv = ''
        self.Y = ''
        self.UY = ''

        m = re.search('U = (.*?) A\^-1 = (.*?) Y = (.*?) U \* Y = (.*)', text)
        if m is not None:
            self.U = parse_matrix(m.group(1))
            self.Ainv = parse_matrix(m.group(2))
            self.Y = parse_matrix(m.group(3))
            self.UY = parse_matrix(m.group(4))

    def __repr__(self):
        out = 'U = \n'
        out += str(self.U)
        out += '\nA^-1 = \n'
        out += str(self.Ainv)
        out += '\nY = \n'
        out += str(self.Y)
        out += '\nUY = \n'
        out += str(self.UY)

        return out

    def is_valid(self):
        if self.U == '' or self.U is None:
            return False
        if self.Ainv == '' or self.U is None:
            return False
        if self.Y == '' or self.U is None:
            return False
        if self.UY == '' or self.U is None:
            return False

        return True


def load_cases(filename):
    with open(filename) as f:
        #Read and put in nice format to apply REGEX
        lines = f.readlines()
        s = ''.join(lines)
        s = s.replace('\n',' ')

        #Find all trap cases
        m = re.findall('UNEQUAL(.*?)TRAPPED',s)

        return [trap_case(i) for i in m]
