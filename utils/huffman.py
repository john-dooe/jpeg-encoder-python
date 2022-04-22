import networkx as nx
from matplotlib import pyplot as plt


class Node:
    def __init__(self):
        self.symbol = None
        self.lchild = None
        self.rchild = None
        self.freq = 0
        self.print_text = ''

    def __lt__(self, other):
        return self.freq < other.freq


class HuffmanTree:
    # 原生哈夫曼编码转为范式哈夫曼编码
    def raw_to_canonical(self, code_dict):
        count = 0
        last_code_len = 0
        last_code = ''
        code_dict_canonical = {}
        for symbol, code in code_dict.items():
            code_len = len(code)
            if count == 0:
                new_code = '0' * code_len
                code_dict_canonical[symbol] = new_code
                last_code = new_code
                last_code_len = code_len
                count += 1
                continue

            if code_len == last_code_len:
                new_code = bin(int(last_code, 2) + 1)[2:].zfill(code_len)
                code_dict_canonical[symbol] = new_code

            else:
                new_code = bin((int(last_code, 2) + 1) << (code_len - last_code_len))[2:].zfill(
                    code_len)
                code_dict_canonical[symbol] = new_code

            last_code = new_code
            last_code_len = code_len
            count += 1

        return code_dict_canonical

    def create_graph(self, graph, node, pos_dict=None, x=0, y=0, layer=1):
        if pos_dict is None:
            pos_dict = {}

        pos_dict[node.print_text] = (x, y)

        if node.lchild is not None:
            graph.add_edge(node.print_text, node.lchild.print_text)
            l_x, l_y = x - 1 / 2 ** layer, y - 1
            l_layer = layer + 1
            self.create_graph(graph, node.lchild, pos_dict, l_x, l_y, l_layer)

        if node.rchild is not None:
            graph.add_edge(node.print_text, node.rchild.print_text)
            r_x, r_y = x + 1 / 2 ** layer, y - 1
            r_layer = layer + 1
            self.create_graph(graph, node.rchild, pos_dict, r_x, r_y, r_layer)

        return graph, pos_dict

    # 哈夫曼树可视化
    def visualize(self, node):
        graph = nx.DiGraph()
        graph, pos_dict = self.create_graph(graph, node)
        fig, ax = plt.subplots(figsize=(12, 8))
        nx.draw_networkx(graph, pos_dict, ax=ax, node_size=300)
        plt.show()


class HuffmanEncoder(HuffmanTree):
    def __init__(self, array):
        self.array = array
        self.freq_dict = self.calc_freq()
        self.root_node = self.build_tree()
        self.code_dict_raw = self.calc_code(self.root_node)
        self.code_dict_raw.pop('eof')
        self.code_dict_raw = dict(sorted(self.code_dict_raw.items(), key=lambda x: len(x[1])))
        self.code_dict = self.raw_to_canonical(self.code_dict_raw)
        # self.visualize(self.root_node)

    # 计算频率
    def calc_freq(self):
        array_len = len(self.array)

        freq_dict = {}
        for symbol in self.array:
            if symbol in freq_dict:
                freq_dict[symbol] += 1
            else:
                freq_dict[symbol] = 1

        # 限制最低出现概率，防止编码长度超过16比特
        min_freq = array_len * (2 ** (-14))
        for symbol, freq in freq_dict.items():
            if freq <= min_freq:
                freq_dict[symbol] = min_freq + 1
        return freq_dict

    # 构建哈夫曼树
    def build_tree(self):
        node_list = []
        for symbol in self.freq_dict:
            node = Node()
            node.symbol = symbol
            node.freq = self.freq_dict[symbol]
            node.print_text = f'{repr(node.symbol)}:{node.freq}'
            node_list.append(node)

        eof_node = Node()
        eof_node.symbol = 'eof'
        eof_node.freq = 0
        eof_node.print_text = f'eof:{0}'
        node_list.append(eof_node)

        count = 0
        while len(node_list) > 1:
            node_list.sort(reverse=True)
            root_node = Node()
            lchild = node_list.pop()
            rchild = node_list.pop()
            root_node.freq = lchild.freq + rchild.freq
            root_node.lchild = lchild
            root_node.rchild = rchild
            root_node.print_text = f'root{count}:{root_node.freq}'
            node_list.append(root_node)
            count += 1

        return node_list[0]

    # 通过频率计算哈夫曼编码
    def calc_code(self, node, code_dict=None, code=''):
        if code_dict is None:
            code_dict = {}

        if node.symbol is not None:
            code_dict[node.symbol] = code
            return
        code += '0'

        if node.lchild is not None:
            self.calc_code(node.lchild, code_dict, code)
        code = code[:-1]
        code += '1'

        if node.rchild is not None:
            self.calc_code(node.rchild, code_dict, code)

        return code_dict

    # 哈夫曼编码原始数据
    def encode(self, array):
        array_encoded = []

        for symbol in array:
            code = self.code_dict[symbol]
            array_encoded.append(code)

        return array_encoded
