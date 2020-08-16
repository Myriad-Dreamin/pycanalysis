import argparse
import multiprocessing
from multiprocessing.util import Finalize
import os
import sys
import random

from typing import Tuple, Union, List

import threading
import graphviz
import re
import functools
import collections

FilteredStringPart = collections.namedtuple('FilteredStringPart', ['filtered', 'offset', 'content'])


# functional helpers


def id_func(x):
    return x


def _compose2(f, g):
    """
    _compose2 return a composed function of f and g, which is represented as `(g \\circ f)` in LaTeX
    :param f: right hand expression of the compose operator
    :param g: left hand expression of the compose operator
    :return: the composed function
    """

    def h(*args, **kwargs):
        u = f(*args, **kwargs)
        # not use instanceof
        if type(u) is tuple:
            return g(*u, **kwargs)
        return g(u, **kwargs)

    return h


def compose(*fs):
    """
    compose return a composed function of a array f(s) reduced by compose operator,
    if the length of fs is 1, then result is just f
    if the length of fs is 0, then raise a exception
    :param fs: array of function to compose
    :return: the composed function
    """
    return functools.reduce(_compose2, fs)


# string related helpers


_validate_id_regex = re.compile(r'^[_a-zA-Z][_a-zA-Z0-9]*$')


def validate_identifier(maybe_id):
    """
    validate that the input of function is a valid c-like identifier using the pattern of _validate_id_regex
    :param maybe_id:
    :return:
    """
    return _validate_id_regex.match(maybe_id) is not None


def match_identifier_r(maybe_bad_id):
    """
    match a valid c-like identifier from suffix as long as possible
    :param maybe_bad_id:
    :return:
    """
    for i, s in enumerate(reversed(maybe_bad_id)):  # type: int, str
        if s == '_' or s.isalnum():
            continue
        else:
            i = len(maybe_bad_id) - i
            while i < len(maybe_bad_id):
                if maybe_bad_id[i].isdigit():
                    i += 1
                else:
                    return maybe_bad_id[i:]
            return ''
    return maybe_bad_id


def get_signature_fn_name_r(source):
    b = 0
    for i, s in enumerate(reversed(source)):
        if s == ')':
            b += 1
        elif s == '(':
            b -= 1
            if b == 0:
                return match_identifier_r(source[:-i - 1].split()[-1])
    return LookupError(f"could not skip the parameter list and get a function name {source}")


def calc_cond(args):
    for i, s in enumerate(args):
        while True:
            if s.startswith('!(!('):
                s = s[4:-2]
                continue
            if s.startswith('!(!'):
                s = s[3:-1]
                continue
            break
        args[i] = s
    return ' && '.join(args)


# dict related helpers


def select_args(ks: List[str], dst: dict = None, src: dict = None):
    if not src or len(src) == 0:
        return
    dst = dst or dict()
    for k in ks:
        if k in src:
            dst[k] = src[k]
    return dst


def get_delete_args(kwargs: dict, k):
    c = kwargs.get(k)
    if c:
        kwargs.pop(k)
    return c


def extend_prefix(prefix, strings):
    return strings + list(map(lambda s: prefix + s, strings))


# console helpers


def str2bool(str_boolean):
    if str_boolean.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str_boolean.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def progressbar(i, total, filename='', size=30, file=sys.stdout):
    x = int(size * i / total)
    to_write = "[%s%s] %i/%i %s" % ("#" * x, "." * (size - x), i, total, filename)
    file.write('\r' + to_write)
    file.flush()


class CommandsProvider(object):
    def __init__(self):
        self.container = dict()

    def count(self, p):
        return p in self.container

    def register(self, p, f):
        if p in self.container:
            raise KeyError('already provided ' + p)
        self.container[p] = f

    def wrap(self, p, wrapper):
        if p not in self.container:
            raise KeyError(p + ' not found')
        self.container[p] = wrapper(self.container[p])

    def invoke(self, p, *args, **kwargs):
        if p not in self.container:
            raise KeyError(p + ' not found')
        self.container[p](*args, **kwargs)


class ArgsHandler(object):

    def __init__(self, commands=None, arg_parser=None, arg_parser_handler=None, **_):
        self.arg_parser = arg_parser or argparse.ArgumentParser()  # type: argparse.ArgumentParser
        self.commands = commands or CommandsProvider()  # type: CommandsProvider
        self.arg_parser_handler = arg_parser_handler
        self.args = None

    @staticmethod
    def should_parse_args(f):
        def wf(self, *a, **kwargs):
            args = self.parse_args()
            d_args = args.__dict__.copy()
            d_args.update(kwargs)
            return f(self, args, *a, **kwargs)

        return wf

    def parse_args(self):
        if self.arg_parser_handler:
            return self.arg_parser_handler()
        if not self.args:
            self.args = self.arg_parser.parse_args()
        return self.args


# render helpers


def yield_color_():
    colors = []

    for i in range(1000):
        colors.append('#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]))

    while True:
        for color in colors:
            yield color


yield_color = yield_color_()


# preprocessors


def preprocess_delete_cross_line_comments(source):
    """
    a easy implementation of preprocess function that delete all cross-line comments in the source code
    for the sake of convenience, we are assuming that no escape asterisk rule is used in the source now
    :param source:
    :return: processed source
    """
    offset = 0
    string_fragments = []
    while True:
        left_most = source.find('/*', offset)
        string_fragments.append(source[offset:left_most])
        if left_most == -1:
            return ''.join(string_fragments)
        first_right = source.find('*/', left_most + 2)
        offset = first_right + 2


def preprocess_delete_inline_line_comments(source):
    """
    a easy implementation of preprocess function that delete all inline comments in the source code
    for the sake of convenience, we are assuming that no escape end line rule is used in the source now
    :param source: the source code text
    :return: processed source
    """
    offset = 0
    string_fragments = []
    while True:
        left_most = source.find('//', offset)
        string_fragments.append(source[offset:left_most])
        if left_most == -1:
            return ''.join(string_fragments)
        first_right = source.find('\n', left_most + 2)
        offset = first_right + 1


def split_source_by_preprocess_commands(source):
    """
    split source code by the preprocess commands
    command \\in '#' { define, undef, ifdef, ifndef, if, else, elif, endif, include, error, pragma }
    :param source: the source code text
    :return: the splitting result
    """
    offset = 0
    string_fragments = []
    while offset < len(source):
        if source[offset] == '#':
            left_most = offset
            width = 1
        else:
            left_most = source.find('\n#', offset)
            width = 2
        if left_most == -1:
            string_fragments.append(FilteredStringPart(False, offset, source[offset:]))
            return string_fragments
        if offset != left_most:
            string_fragments.append(FilteredStringPart(False, offset, source[offset:left_most]))
        fixed = left_most + width
        while True:
            first_right = source.find('\n', fixed)
            if first_right == -1:
                string_fragments.append(FilteredStringPart(True, left_most + width - 1, source[left_most:]))
                return string_fragments
            if source[first_right - 1] == '\\':
                fixed = first_right + 1
                continue
            break

        string_fragments.append(FilteredStringPart(True, left_most + width - 1, source[left_most:first_right + 1]))
        offset = first_right + 1
    return string_fragments


class CallEdge:
    def __init__(self, u, v, condition=''):
        self.u = u
        self.v = v
        self.condition = condition

    def __hash__(self):
        return hash(f'{self.u}${self.v}${self.condition}')

    def __eq__(self, other):
        return f'{self.u}${self.v}${self.condition}' == f'{other.u}${other.v}${other.condition}'


class PreprocessorCommandParser(object):
    _state_macro_nothing = 0
    _state_macro_if = 1
    _state_macro_else = 2

    def __init__(self):
        self.nodes, self.edges = None, None  # type: Union[set, None], Union[set, None]
        self.block_embedded = None  # type: Union[int, None]
        self.not_parsed_source, self.signature_fn_name = None, None  # type: Union[str, None], Union[str, None]
        self.cond, self.cond_stack = None, None  # type: Union[list, None], Union[list, None]
        self.macro_state, self.macro_state_stack = None, None  # type: Union[int, None], Union[list, None]

    def reset(self):
        self.nodes, self.edges = set(), set()
        self.block_embedded = 0
        self.not_parsed_source, self.signature_fn_name = '', ''
        self.cond, self.cond_stack = [], []
        self.macro_state, self.macro_state_stack = 0, []

    def build_callgraph(self, parts, offset_mapper=None):
        _ = offset_mapper  # will add this feature later
        self.reset()

        for index, part in enumerate(parts):  # type: int, Tuple[bool, int, str]
            filtered, offset, content = part
            if filtered:
                self.parse_preprocess_command(content)
            else:
                self.parse_c_code(content)
        return self.nodes, self.edges

    def parse_preprocess_command(self, content):
        stripped_contend = content.strip()
        last_if = True
        if stripped_contend.startswith('#ifdef'):
            self.cond_stack.append(self.cond)
            self.cond = [stripped_contend[6:].strip()]
            self.macro_state_stack.append((self.block_embedded, self.macro_state))
            self.macro_state = PreprocessorCommandParser._state_macro_if
        elif stripped_contend.startswith('#if'):
            self.cond_stack.append(self.cond)
            self.cond = [stripped_contend[3:].strip()]
            self.macro_state_stack.append((self.block_embedded, self.macro_state))
        else:
            eif = re.search(r'#el(?:se\s+)?if', stripped_contend)
            if eif:
                self.cond = self.cond[:-1] + [
                    f'!({self.cond[-1]})', stripped_contend[len(eif[0]):].strip()]
                self.macro_state = PreprocessorCommandParser._state_macro_else
                last_block_embedded = self.macro_state_stack[-1][0]
                self.block_embedded = last_block_embedded
            last_if = eif
        if not last_if and stripped_contend.startswith('#else'):
            # if len(self.cond_stack) == 0: raise
            self.cond = self.cond[:-1] + [f'!({self.cond[-1]})']
            self.macro_state = PreprocessorCommandParser._state_macro_else
            last_block_embedded = self.macro_state_stack[-1][0]
            self.block_embedded = last_block_embedded
        elif stripped_contend.startswith('#endif'):
            self.cond = self.cond_stack.pop()
            if len(self.macro_state_stack):
                last_block_embedded, self.macro_state = self.macro_state_stack.pop()
                # self.block_embedded = last_block_embedded

    def parse_c_code(self, content):
        # parse self.not_parsed_source
        exploded = re.split('([{}])', content)
        for e in exploded:
            if e == '{':
                self.block_embedded += 1
                if self.block_embedded == 1:
                    self.try_match_a_source()
            elif e == '}':
                self.block_embedded -= 1
                if self.block_embedded == 0:
                    self.signature_fn_name = ''
            elif self.block_embedded == 0:
                self.not_parsed_source += e
            elif len(self.signature_fn_name) != 0:
                self.add_call_edges(e)

    def try_match_a_source(self):
        self.signature_fn_name = get_signature_fn_name_r(self.not_parsed_source)
        if isinstance(self.signature_fn_name, Exception):
            if not type(self.signature_fn_name) is LookupError:
                raise self.signature_fn_name
            self.signature_fn_name = ''
        else:
            self.nodes.add(self.signature_fn_name)
        self.not_parsed_source = ''

    def add_call_edges(self, content):
        for statements in content.split(';'):
            for target in list(map(
                    # skip empty string
                    lambda maybe_callee: (maybe_callee.split()[-1] if len(maybe_callee.strip()) else maybe_callee),
                    # break on left parentheses
                    statements.split('(')[:-1])):

                # skip mismatched keywords
                if target.strip() in ['if', '', 'while', 'for']:
                    continue

                if validate_identifier(target):
                    self.edges.add(CallEdge(u=self.signature_fn_name, v=target, condition=calc_cond(self.cond)))
                    self.nodes.add(target)


# call graph builder


class CallGraphBuilder(object):
    @staticmethod
    def get_build_args(args: dict):
        return select_args(extend_prefix('builder_', ['pc_parser']), src=args)

    @staticmethod
    def build(source, pc_parser=None):
        """
        parse a source to call graph
        :param source:
        :param pc_parser:
        :return:
        """
        parts = split_source_by_preprocess_commands(compose(
            preprocess_delete_cross_line_comments,
            preprocess_delete_inline_line_comments,
        )(source))
        # source = functools.reduce(lambda u, v: u if v.filtered else u + v.content, parts, '')
        if not pc_parser:
            pc_parser = PreprocessorCommandParser()
        return pc_parser.build_callgraph(parts)


# graphviz render


class GraphvizRender(object):
    @staticmethod
    def get_digraph_args(args: dict):
        return select_args(extend_prefix('digraph_', ['comment']), src=args)

    @staticmethod
    def get_render_args(args: dict):
        return select_args(extend_prefix('render_', ['view', 'filename', 'directory', 'cleanup']), src=args)

    @staticmethod
    def add_target_directory_option(args: dict, target):
        args['filename'] = os.path.basename(target)
        args['directory'] = os.path.dirname(target)
        return args

    @staticmethod
    def render_(nodes, edges, digraph_args, render_args):

        G = graphviz.Digraph(**digraph_args)
        # G.graph_attr.update(ratio='1.618', size='7 x 10')
        for node in nodes:
            G.node(node)
        for edge in edges:
            G.edge(edge.u, edge.v, label=edge.condition, color=next(yield_color))
        G.render(**render_args)

    @staticmethod
    def render(nodes, edges, **kwargs):
        r = GraphvizRender
        return r.render_(nodes, edges, digraph_args=r.get_digraph_args(kwargs), render_args=r.get_render_args(kwargs))


# static file render functions


def render_structured_info(vertices, edges, renderer, digraph_args=None, render_args=None,
                           add_target_directory_option=None, target=None):
    return renderer.render_(
        vertices, edges, digraph_args=digraph_args,
        render_args=render_args if target is None else add_target_directory_option(render_args, target))


def render_source(code, builder, renderer,
                  builder_args=None, digraph_args=None, render_args=None,
                  add_target_directory_option=None, target=None):
    V, E = builder.build(code, **builder_args)
    render_structured_info(V, E, renderer, digraph_args, render_args, add_target_directory_option, target)
    return V, E


def render_source_with_profile(*args, **kwargs):
    global local_profile
    local_profile.enable()
    res = render_source(*args, **kwargs)
    local_profile.disable()
    return res


def render_file(filepath, *options, **kwargs):
    return render_source(open(filepath).read(), *options, **kwargs)


def render_file_with_profile(filepath, *options, **kwargs):
    return render_source_with_profile(open(filepath).read(), *options, **kwargs)


local_profile = None


def __subprocess_profile_initializer():
    global local_profile
    import cProfile
    local_profile = cProfile.Profile()

    def fin():
        local_profile.dump_stats('profile-%s.out' % multiprocessing.current_process().pid)

    Finalize(None, fin, exitpriority=1)


def default_add_target_directory_option(args, target):
    args['dst'] = target
    return args


def apply_render_call_graph(filepath, builder=CallGraphBuilder, renderer=GraphvizRender,
                            dst='build', progress_bar=None, processes=None, profile=None, **kwargs):
    add_target_directory_option = getattr(renderer, 'add_target_directory_option', None) or kwargs.get(
        'add_target_directory_option', default_add_target_directory_option)

    builder_args = getattr(builder, 'get_build_args', id_func)(kwargs)
    digraph_args = getattr(renderer, 'get_digraph_args', id_func)(kwargs)
    render_args = getattr(renderer, 'get_render_args', id_func)(kwargs)

    options = (builder, renderer, builder_args, digraph_args, render_args, add_target_directory_option)

    if profile:
        rs = render_source_with_profile
        rf = render_file_with_profile
    else:
        rs = render_source
        rf = render_file

    source = kwargs.get('source')
    if source is not None:
        render_args.update({'view': True, 'cleanup': True})
        rs(source, *options)
        return

    if isinstance(filepath, str):
        if os.path.isfile(filepath):
            rs(open(filepath).read(), *options, target=os.path.join(dst, os.path.basename(filepath)))
            return
        elif os.path.isdir(filepath):
            source_list = []
            for rt, dirs, files in os.walk(filepath):
                _ = dirs
                for f in files:
                    if f.endswith('.c') or f.endswith('.h'):
                        source_list.append((rt, f))
        else:
            raise ValueError('want filepath value is file or directory')
    elif isinstance(filepath, list):
        source_list = filepath
    else:
        raise TypeError('want filepath type is str or list')

    if processes:
        initializer = None
        if profile:
            initializer = __subprocess_profile_initializer
        pool = multiprocessing.Pool(processes=processes, initializer=initializer)

        def create_work(work_func, *args, **kwargs2):
            return pool.apply_async(work_func, args, kwds=kwargs2, callback=get_delete_args(kwargs2, 'work_callback'))

        def start_work(work_thread):
            return work_thread.get()

        def clean_resources():
            pool.close()
            pool.join()
    else:
        def create_work(work_func, *args, **kwargs2):
            c = get_delete_args(kwargs2, 'work_callback')

            def lambda_with_cb():
                res = work_func(*args, **kwargs2)
                c and c(res)

            return lambda_with_cb

        def start_work(lazy_func):
            return lazy_func()

        def clean_resources():
            pass

    sl = len(source_list)
    works = []
    for i, s in enumerate(map(lambda x: os.path.join(*x), source_list)):
        works.append((i, s, create_work(rf, s, *options, os.path.join(dst, s))))
    for i, s, w in works:
        start_work(w)
        progress_bar and progress_bar(i, sl, s)
    clean_resources()
    progress_bar and progress_bar(sl, sl)


class DrawController(ArgsHandler):
    cmd_name = 'draw'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ap = self.arg_parser

        ap.add_argument('--src', required=True, help='Source Path')
        ap.add_argument('--dst', default=kwargs.get('dst', 'build'), nargs='?', help='Output Path')
        ap.add_argument('--merge', type=str2bool, nargs='?', const=True, default=False,
                        help='Helps FullView')

        self.commands.register('apply_render_call_graph', self.apply_render_call_graph)
        self.commands.register('render_structured_info', self.render_structured_info)

        ap.set_defaults(func=self)

    def __call__(self, src=None, **kwargs):
        args = self.parse_args()
        isMerge, VM, EM = args.merge, None, None
        if isMerge:
            VM, EM = set(), set()
            kwargs['work_callback'] = kwargs.get('work_callback', lambda ves: (VM.update(ves[0]), EM.update(ves[1])))
        self.apply_render_call_graph(src, **kwargs)
        if isMerge:
            self.render_structured_info(VM, EM, **kwargs)
        return self

    @ArgsHandler.should_parse_args
    def apply_render_call_graph(self, args, src=None, **kwargs):
        return apply_render_call_graph(src or args.src, **kwargs)

    @ArgsHandler.should_parse_args
    def render_structured_info(self, args, vertices, edges, renderer=GraphvizRender, target=None, **kwargs):
        add_target_directory_option = getattr(renderer, 'add_target_directory_option', None) or kwargs.get(
            'add_target_directory_option', default_add_target_directory_option)

        digraph_args = getattr(renderer, 'get_digraph_args', id_func)(kwargs)
        render_args = getattr(renderer, 'get_render_args', id_func)(kwargs)
        return render_structured_info(vertices, edges, renderer, digraph_args, render_args, add_target_directory_option,
                                      target=target or os.path.join(args.dst, 'full_graph.gv'))


class ViewController(ArgsHandler):
    cmd_name = 'view'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ap = self.arg_parser

        ap.add_argument('viewing_function', help='source function')
        ap.add_argument('--max_depth', type=int, default=2, help='search depth')

        ap.set_defaults(func=self)

    @ArgsHandler.should_parse_args
    def __call__(self, args, renderer=GraphvizRender(), progress_bar=None, **kwargs):
        vertices = {args.viewing_function}
        edges = set()

        import json
        with open('.parsed.json') as f:
            d = json.load(f)
        edges_next = dict()
        edges_last = dict()
        for edge in d['edges']:
            ol = edges_next.get(edge['front'], list())
            ol.append(edge)
            edges_next[edge['front']] = ol
            ol = edges_last.get(edge['tail'], list())
            ol.append(edge)
            edges_last[edge['tail']] = ol

        max_depth = args.max_depth - 1

        def sort_add(edges_set, direction, swap):
            queue = collections.deque()
            queue.append((args.viewing_function, 0))
            progress_a, progress_b = 0, 1
            added = {args.viewing_function}
            while len(queue):
                progress_bar and progress_bar(progress_a, progress_b)
                progress_a += 1
                f, depth = queue.pop()
                fs = edges_set.get(f)
                if not fs:
                    continue
                for e in fs:
                    tail = e[direction]
                    vertices.add(tail)
                    u, v = swap(f, tail)
                    edges.add(CallEdge(u=u, v=v, condition=e.get('label')))
                    if tail not in added and depth < max_depth:
                        added.add(tail)
                        queue.append((tail, depth + 1))
                        progress_b += 1
            progress_bar and progress_bar(progress_b, progress_b)

        sort_add(edges_next, 'tail', lambda u, v: (u, v))
        print()
        sort_add(edges_last, 'front', lambda u, v: (v, u))
        print()
        print('found', len(vertices), 'vertices', len(edges), 'edges')
        import tempfile, uuid
        kwargs['filename'] = 'view-digraph-' + str(uuid.uuid4())[-12:] + '.gv'
        kwargs['directory'] = tempfile.gettempdir()
        kwargs['view'] = True
        kwargs['cleanup'] = True

        digraph_args = getattr(renderer, 'get_digraph_args', id_func)(kwargs)
        render_args = getattr(renderer, 'get_render_args', id_func)(kwargs)
        return render_structured_info(vertices, edges, renderer, digraph_args, render_args)


_ParenthesesNothing = 0
_ParenthesesParen = 1
_ParenthesesBrace = 2
_ParenthesesBracket = 3


class ParenthesesParser(object):

    def __init__(self, source):
        self.i = 0
        self.source = source
        self.parsed = []

    def add_not_parsed(self, li):
        stripped = self.source[li:self.i].strip()
        if len(stripped):
            self.parsed.extend(map(lambda x: (_ParenthesesNothing, x.strip()), stripped.split('\n')))

    def parse_parentheses(self, li, mark, lp, rp):
        if self.source[self.i] == lp:
            self.add_not_parsed(li)
            self.i += 1
            p = self.parsed
            self.parsed = []
            self.work(rp)
            if self.source[self.i] != rp:
                raise ValueError('not wanted')
            p.append((mark, self.parsed))
            self.parsed = p
            return self.i + 1
        return None

    def work(self, want_matched=None):
        last_index = self.i
        while self.i < len(self.source):
            if self.source[self.i] == want_matched:
                self.add_not_parsed(last_index)
                return
            last_index = \
                self.parse_parentheses(last_index, _ParenthesesBrace, '{', '}') or \
                self.parse_parentheses(last_index, _ParenthesesBracket, '[', ']') or \
                self.parse_parentheses(last_index, _ParenthesesParen, '(', ')') or \
                last_index
            self.i += 1
        self.add_not_parsed(last_index)


def parse_parentheses(source):
    p = ParenthesesParser(source)
    p.work()
    return p.parsed


def m_range(x):
    for _r in range(x):
        yield _r


class Digraph(object):
    regex = re.compile(r'\s*([a-zA-Z0-9]*)="([^"]*)"')

    def __init__(self, parsed):
        self.nodes, self.edges = list(), list()

        r = m_range(len(parsed))
        for i in r:
            t, c = parsed[i]
            if t == _ParenthesesNothing:
                if c == '""':
                    continue
                elif c == "digraph":
                    raise NotImplementedError("not recursive now")
                else:
                    if '->' in c:
                        a, b = c.split('->')
                        edge = {'front': a.strip(), 'tail': b.strip()}
                        if parsed[i + 1][0] == _ParenthesesBracket:
                            next(r)
                            options = parsed[i + 1][1][0][1]
                            for matched in Digraph.regex.findall(options):
                                edge[matched[0]] = matched[1]
                        self.edges.append(edge)
                    else:
                        print('node', c)
                        if parsed[i + 1][0] == _ParenthesesBracket:
                            next(r)
                            raise NotImplementedError("...")
                        self.nodes.append(c)


class ConvertController(ArgsHandler):
    cmd_name = 'convert'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ap = self.arg_parser

        ap.add_argument('--format', '-j', default=kwargs.get('convert_format', 'json'),
                        nargs='?', help='Format')
        ap.add_argument('--src', required=True, help='Source Path')

        # subparsers = ap.add_subparsers()
        #
        # self.draw_controller = Controller.Draw(
        #     arg_parser=subparsers.add_parser(Controller.Draw.cmd_name), arg_parser_handler=self.parse_args, **kwargs)

        ap.set_defaults(func=self)

    def __call__(self, *args, **kwargs):
        return self.convert_digraph(*args, **kwargs)

    @ArgsHandler.should_parse_args
    def convert_digraph(self, args, **kwargs):
        s = open(args.src).read()
        parsed = parse_parentheses(s)
        r = m_range(len(parsed))
        structured = []
        for i in r:
            t, c = parsed[i]
            if t == _ParenthesesNothing:
                if c == "digraph":
                    i = next(r)
                    t, c = parsed[i]
                    if t != _ParenthesesBrace:
                        raise ValueError("not matched {")
                    structured.append(Digraph(c))

        if len(structured) != 1:
            raise IndexError("not 1")
        g = structured[0]
        if type(g) != Digraph:
            raise TypeError("want parsed is digraph")
        d = {'nodes': g.nodes, 'edges': g.edges}
        import json
        with open('.parsed.json', 'w') as f:
            json.dump(d, f)


class Controller(ArgsHandler):
    Draw = DrawController
    Convert = ConvertController
    View = ViewController

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ap = self.arg_parser
        ap.add_argument('--processes', '-j', default=kwargs.get('processes'),
                        nargs='?', help='Core Use Count')
        ap.add_argument('--profile', type=str2bool, nargs='?', const=True, default=False,
                        help='Helps Profile')

        subparsers = ap.add_subparsers()

        self.draw_controller = Controller.Draw(
            arg_parser=subparsers.add_parser(Controller.Draw.cmd_name), arg_parser_handler=self.parse_args, **kwargs)
        self.convert_controller = Controller.Convert(
            arg_parser=subparsers.add_parser(Controller.Convert.cmd_name), arg_parser_handler=self.parse_args, **kwargs)
        self.view_controller = Controller.View(
            arg_parser=subparsers.add_parser(Controller.View.cmd_name), arg_parser_handler=self.parse_args, **kwargs)

    def __call__(self, *a, **kwargs):
        args = self.parse_args()
        if 'func' not in args:
            self.arg_parser.print_help()
            return

        isProfiling, pr = args.profile, None
        if isProfiling:
            import cProfile
            pr = cProfile.Profile()
            pr.enable()

        args.func(*a, **kwargs)

        if isProfiling:
            pr.disable()
            pr.dump_stats('profile.pstat')


if __name__ == '__main__':
    # sample 'drivers/base/dma_buf_lock/src/dma_buf_lock.c'
    # def pb(*args, **kwargs):
    #     return progressbar(*args, **kwargs, file=sys.stderr)
    mutex = threading.Lock()


    def safe_progressbar(*args, **kwargs):
        # mutex.acquire()
        progressbar(*args, **kwargs)
        # mutex.release()


    Controller(dst='build', processes=multiprocessing.cpu_count())(
        progress_bar=safe_progressbar)
