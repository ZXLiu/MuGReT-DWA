import os
import json
import javalang
import javalang.tree
import javalang.ast
from tqdm import tqdm
from javalang.ast import Node
from anytree import AnyNode, RenderTree


JDK_WHITELIST = {
        # 核心类
        'System', 'String', 'Object', 'Math', 'Thread', 'Exception', 'Throwable',
        'Integer', 'Double', 'Float', 'Long', 'Boolean', 'Byte', 'Short', 'Character', 'Void',
        'StringBuilder', 'StringBuffer', 'Class', 'Runtime', 'Process', 'Enum',
        'List', 'ArrayList', 'LinkedList', 'Map', 'HashMap', 'TreeMap', 'Set', 'HashSet',
        'Collections', 'Arrays', 'Iterator', 'Iterable', 'Stream', 'Optional',
        'File', 'Path', 'Paths', 'Files', 'InputStream', 'OutputStream', 'Reader', 'Writer',
        'Scanner', 'Pattern', 'Matcher', 'Date', 'Calendar', 'Timer', 'Random',
        # 常见异常
        'RuntimeException', 'IOException', 'NullPointerException', 'IndexOutOfBoundsException',
        'ClassNotFoundException', 'NoSuchMethodException', 'InterruptedException',
        'IllegalArgumentException', 'IllegalStateException', 'StringIndexOutOfBoundsException',
        # 核心方法与属性
        'println', 'print', 'printf', 'format', 'out', 'in', 'err', 'exit',
        'length', 'size', 'isEmpty', 'get', 'set', 'add', 'remove', 'clear',
        'toString', 'equals', 'hashCode', 'getClass', 'clone', 'substring', 'trim',
        'toLowerCase', 'toUpperCase', 'split', 'replace', 'indexOf', 'charAt',
        'valueOf', 'parseInt', 'parseDouble', 'main', 'run', 'start', 'close', 'flush',
        'hasNext', 'next', 'append', 'max', 'min', 'abs', 'pow', 'sqrt', 'stream', 'collect'
    }

JAVA_KEYWORDS = ['public', 'private', 'protected', 'static', 'final', 'class', 'interface', 'if', 'else', 'for', 'while', 'return', 'new', 'try', 'catch', 'throw']

edges={'Nexttoken':2,'Prevtoken':3,'Nextuse':4,'Prevuse':5,'If':6,'Ifelse':7,'While':8,'For':9,'Nextstmt':10,'Prevstmt':11,'Prevsib':12}

forbideen_files = ['37044', '4892654', '6966398', '7550876']


def load_new_add_data():
    dir_path = "/home/liutongxue/PycharmProject/MuGReT-DWA/MuGReT/data/data_source/dataset_bigclonebench"
    label_path = os.path.join(dir_path,"new_clone_labels.txt")
    with open(label_path, 'r') as f:
        labels = f.readlines()
        labels = [label.strip().split(',') for label in labels]
    code_files = []
    for ele in labels:
        code_files.append(ele[0] + '.java')
        code_files.append(ele[1] + '.java')
    return code_files, labels


def normalize_ast(node):
    """
    通用 AST 规范化函数:
    """

    JDK_WHITELIST = {
        # 核心类
        'System', 'String', 'Object', 'Math', 'Thread', 'Exception', 'Throwable',
        'Integer', 'Double', 'Float', 'Long', 'Boolean', 'Byte', 'Short', 'Character', 'Void',
        'StringBuilder', 'StringBuffer', 'Class', 'Runtime', 'Process', 'Enum',
        'List', 'ArrayList', 'LinkedList', 'Map', 'HashMap', 'TreeMap', 'Set', 'HashSet',
        'Collections', 'Arrays', 'Iterator', 'Iterable', 'Stream', 'Optional',
        'File', 'Path', 'Paths', 'Files', 'InputStream', 'OutputStream', 'Reader', 'Writer',
        'Scanner', 'Pattern', 'Matcher', 'Date', 'Calendar', 'Timer', 'Random',
        # 常见异常
        'RuntimeException', 'IOException', 'NullPointerException', 'IndexOutOfBoundsException',
        'ClassNotFoundException', 'NoSuchMethodException', 'InterruptedException',
        'IllegalArgumentException', 'IllegalStateException', 'StringIndexOutOfBoundsException',
        # 核心方法与属性
        'println', 'print', 'printf', 'format', 'out', 'in', 'err', 'exit',
        'length', 'size', 'isEmpty', 'get', 'set', 'add', 'remove', 'clear',
        'toString', 'equals', 'hashCode', 'getClass', 'clone', 'substring', 'trim',
        'toLowerCase', 'toUpperCase', 'split', 'replace', 'indexOf', 'charAt',
        'valueOf', 'parseInt', 'parseDouble', 'main', 'run', 'start', 'close', 'flush',
        'hasNext', 'next', 'append', 'max', 'min', 'abs', 'pow', 'sqrt', 'stream', 'collect'
    }

    maps = {'type': {}, 'func': {}, 'var': {}, 'const': {}}
    counters = {'type': 1, 'func': 1, 'var': 1, 'const': 1}

    # === 【关键】判断是否为 JDK 限定符 ===
    def is_jdk_qualifier(qualifier_str):
        if not qualifier_str:
            return False

        # 情况 1: 完全匹配 (例如 "String")
        if qualifier_str in JDK_WHITELIST:
            return True

        # 情况 2: 复合结构 (例如 "System.out", "Math.PI", "Integer.MAX_VALUE")
        # 逻辑：如果开头的类名 (System) 是 JDK 的，那后面的属性通常也是 JDK 的
        if "." in qualifier_str:
            root = qualifier_str.split('.')[0]
            if root in JDK_WHITELIST:
                return True

        return False

    def get_mapped_name(original_name, type_key, prefix):
        # 白名单检查
        if original_name in JDK_WHITELIST:
            return original_name

        if original_name not in maps[type_key]:
            new_name = f"{prefix}{counters[type_key]}"
            maps[type_key][original_name] = new_name
            counters[type_key] += 1
        return maps[type_key][original_name]

    def walk_and_rename(n):
        if not n: return

        # --- 1. 类与类型 ---
        if isinstance(n, (javalang.tree.ClassDeclaration, javalang.tree.InterfaceDeclaration)):
            if n.name: n.name = get_mapped_name(n.name, 'type', 't')
        elif isinstance(n, javalang.tree.ReferenceType):
            if n.name: n.name = get_mapped_name(n.name, 'type', 't')

        # --- 2. 方法与调用 ---
        elif isinstance(n, javalang.tree.MethodDeclaration):
            if n.name: n.name = get_mapped_name(n.name, 'func', 'f')
        elif isinstance(n, javalang.tree.MethodInvocation):
            if n.member: n.member = get_mapped_name(n.member, 'func', 'f')

            # 【修改点】处理 qualifier (如 System.out)
            if n.qualifier:
                if is_jdk_qualifier(n.qualifier):
                    pass  # 是 System.out 或 Math，保持原样
                else:
                    n.qualifier = get_mapped_name(n.qualifier, 'var', 'v')

        # --- 3. 变量与引用 ---
        elif isinstance(n, (javalang.tree.VariableDeclarator, javalang.tree.FormalParameter)):
            if n.name: n.name = get_mapped_name(n.name, 'var', 'v')
        elif isinstance(n, javalang.tree.MemberReference):
            if n.member: n.member = get_mapped_name(n.member, 'var', 'v')

            # 【修改点】处理 qualifier (如 System.out)
            if n.qualifier:
                if is_jdk_qualifier(n.qualifier):
                    pass  # 保持原样
                else:
                    n.qualifier = get_mapped_name(n.qualifier, 'var', 'v')

        # --- 4. 常量 ---
        elif isinstance(n, javalang.tree.Literal):
            if n.value and n.value not in ['true', 'false', 'null']:
                n.value = get_mapped_name(n.value, 'const', 'c')

        # 递归
        if hasattr(n, 'children'):
            for child in n.children:
                if isinstance(child, list):
                    for item in child:
                        if isinstance(item, javalang.ast.Node): walk_and_rename(item)
                elif isinstance(child, javalang.ast.Node):
                    walk_and_rename(child)

    walk_and_rename(node)
    return node


def create_ast(dir_path, label_path, dataset, save_vocab_path=None):

    with open(label_path, 'r') as f:
        labels = f.readlines()
        labels = [label.strip().split(',') for label in labels]

    filtered_labels = []
    for ele in labels:
        dataset_lable = int(ele[4])
        if dataset=="BigCloneBench" and dataset_lable == 0:
            if str(ele[1]) in forbideen_files or str(ele[0]) in forbideen_files:
                continue
            filtered_labels.append(ele)
        elif dataset=="GoogleCodeJam" and dataset_lable==1:
            filtered_labels.append(ele)
    
    code_files = []
    for ele in filtered_labels:
        code_files.append(ele[0] + '.java')
        code_files.append(ele[1] + '.java')

    add_new_code_files, add_new_labels = load_new_add_data()

    # filtered_labels += add_new_labels
    code_files += add_new_code_files
    code_files = sorted(set(code_files))

    asts=[]
    paths=[]
    alltokens=[]
    for file in tqdm(code_files, total=len(code_files)):
        programfile=open(os.path.join(dir_path,file),encoding='utf-8')
        programtext=programfile.read()
        try:
            # 1. 词法分析 (Lexer) -> 这里可能报 LexerError
            programtokens=javalang.tokenizer.tokenize(programtext)
            # 2. 语法分析 (Parser) -> 这里可能报 JavaSyntaxError
            parser=javalang.parse.Parser(programtokens)
            programast=parser.parse_member_declaration()
            # 3. 规范化
            normalize_ast(programast)
        except (javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError, Exception) as e:
            print(f"Error parsing {file}: {e}")
            continue

        paths.append(os.path.join(dir_path,file))
        asts.append(programast)
        get_sequence(programast,alltokens)
        programfile.close()

    if save_vocab_path and not os.path.exists(save_vocab_path):
        base_tokens = list(JDK_WHITELIST) + JAVA_KEYWORDS
        # base_tokens = list(JDK_WHITELIST) + PRE_ALLOCATED_TOKENS + JAVA_KEYWORDS
        alltokens = list(set(base_tokens + alltokens))
        astdict = dict(zip(paths, asts))
        vocabsize = len(alltokens)
        tokenids = range(vocabsize)
        vocabdict = dict(zip(alltokens, tokenids))

        try:
            with open(save_vocab_path, 'w', encoding='utf-8') as f:
                json.dump(vocabdict, f, indent=4, ensure_ascii=False)
            print(f"Vocab saved to: {save_vocab_path}")
        except Exception as e:
            print(f"Error saving vocab: {e}")
    elif save_vocab_path and os.path.exists(save_vocab_path):
        astdict = dict(zip(paths, asts))
        with open(save_vocab_path, 'r', encoding='utf-8') as f:
            vocabdict = json.load(f)
        vocabsize = len(vocabdict)

    return astdict, vocabsize, vocabdict, filtered_labels, add_new_labels


def create_ast_eval(dataset, save_vocab_path):
    asts=[]
    index=[]
    for item in tqdm(dataset, total=len(dataset)):
        programtext = item["fixed_function"]
        try:
            # 1. 词法分析 (Lexer) -> 这里可能报 LexerError
            programtokens=javalang.tokenizer.tokenize(programtext)
            # 2. 语法分析 (Parser) -> 这里可能报 JavaSyntaxError
            parser=javalang.parse.Parser(programtokens)
            programast=parser.parse_member_declaration()
            # 3. 规范化
            normalize_ast(programast)
        except (javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError, Exception) as e:
            print(f"Error parsing {item['index']}: {e}")
            continue

        index.append(item["index"])
        asts.append(programast)
    astdict = dict(zip(index, asts))

    with open(save_vocab_path, 'r', encoding='utf-8') as f:
        vocabdict = json.load(f)
    vocabsize = len(vocabdict)

    return astdict, vocabsize, vocabdict


def create_ast_single(data):
    try:
        # 1. 词法分析 (Lexer) -> 这里可能报 LexerError
        programtokens=javalang.tokenizer.tokenize(data)
        # 2. 语法分析 (Parser) -> 这里可能报 JavaSyntaxError
        parser=javalang.parse.Parser(programtokens)
        programast=parser.parse_member_declaration()
        # 3. 规范化
        normalize_ast(programast)
        return programast
    except (javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError, Exception) as e:
        print(f"Error parsing: {e}")
        return None


def createseparategraph(astdict,vocabdict,mode='astonly',nextsib=False,ifedge=False,whileedge=False,foredge=False,blockedge=False,nexttoken=False,nextuse=False):
    pathlist=[]
    treelist=[]
    graph_dict = {}
    for path,tree in astdict.items():
        #print(tree)
        #print(path)
        nodelist = []
        newtree=AnyNode(id=0,token=None,data=None)
        createtree(newtree, tree, nodelist)
        #print(path)
        #print(newtree)
        x = []
        edgesrc = []
        edgetgt = []
        edge_attr=[]
        if mode=='astonly':
            getnodeandedge_astonly(newtree, x, vocabdict, edgesrc, edgetgt)
        else:
            getnodeandedge(newtree, x, vocabdict, edgesrc, edgetgt,edge_attr)
            if nextsib==True:
                getedge_nextsib(newtree,vocabdict,edgesrc,edgetgt,edge_attr)
            getedge_flow(newtree,vocabdict,edgesrc,edgetgt,edge_attr,ifedge,whileedge,foredge)
            if blockedge==True:
                getedge_nextstmt(newtree,vocabdict,edgesrc,edgetgt,edge_attr)
            tokenlist=[]
            if nexttoken==True:
                getedge_nexttoken(newtree,vocabdict,edgesrc,edgetgt,edge_attr,tokenlist)
            variabledict={}
            if nextuse==True:
                getedge_nextuse(newtree,vocabdict,edgesrc,edgetgt,edge_attr,variabledict)
        edge_index=[edgesrc, edgetgt]
        astlength=len(x)
        pathlist.append(path)
        file_name = os.path.splitext(os.path.basename(path))[0]
        treelist.append([[x,edge_index,edge_attr],astlength])
        graph_dict[file_name]=[[x,edge_index,edge_attr],astlength]

    return graph_dict


def getnodeandedge_astonly(node,nodeindexlist,vocabdict,src,tgt):
    token=node.token
    nodeindexlist.append([vocabdict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        src.append(child.id)
        tgt.append(node.id)
        getnodeandedge_astonly(child,nodeindexlist,vocabdict,src,tgt)


def getnodeandedge(node,nodeindexlist,vocabdict,src,tgt,edgetype):
    token=node.token
    nodeindexlist.append([vocabdict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        edgetype.append([0])
        src.append(child.id)
        tgt.append(node.id)
        edgetype.append([0])
        getnodeandedge(child,nodeindexlist,vocabdict,src,tgt,edgetype)


def getedge_flow(node,vocabdict,src,tgt,edgetype,ifedge=False,whileedge=False,foredge=False):
    token=node.token
    if whileedge==True:
        if token=='WhileStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([edges['While']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([edges['While']])
    if foredge==True:
        if token=='ForStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([edges['For']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([edges['For']])
    if ifedge==True:
        if token=='IfStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([edges['If']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([edges['If']])
            if len(node.children)==3:
                src.append(node.children[0].id)
                tgt.append(node.children[2].id)
                edgetype.append([edges['Ifelse']])
                src.append(node.children[2].id)
                tgt.append(node.children[0].id)
                edgetype.append([edges['Ifelse']])
    for child in node.children:
        getedge_flow(child,vocabdict,src,tgt,edgetype,ifedge,whileedge,foredge)


def getedge_nextstmt(node,vocabdict,src,tgt,edgetype):
    token=node.token
    if token=='BlockStatement':
        for i in range(len(node.children)-1):
            src.append(node.children[i].id)
            tgt.append(node.children[i+1].id)
            edgetype.append([edges['Nextstmt']])
            src.append(node.children[i+1].id)
            tgt.append(node.children[i].id)
            edgetype.append([edges['Prevstmt']])
    for child in node.children:
        getedge_nextstmt(child,vocabdict,src,tgt,edgetype)


def getedge_nexttoken(node,vocabdict,src,tgt,edgetype,tokenlist):
    def gettokenlist(node,vocabdict,edgetype,tokenlist):
        token=node.token
        if len(node.children)==0:
            tokenlist.append(node.id)
        for child in node.children:
            gettokenlist(child,vocabdict,edgetype,tokenlist)
    gettokenlist(node,vocabdict,edgetype,tokenlist)
    for i in range(len(tokenlist)-1):
            src.append(tokenlist[i])
            tgt.append(tokenlist[i+1])
            edgetype.append([edges['Nexttoken']])
            src.append(tokenlist[i+1])
            tgt.append(tokenlist[i])
            edgetype.append([edges['Prevtoken']])


def getedge_nextuse(node,vocabdict,src,tgt,edgetype,variabledict):
    def getvariables(node,vocabdict,edgetype,variabledict):
        token=node.token
        if token=='MemberReference':
            for child in node.children:
                if child.token==node.data.member:
                    variable=child.token
                    variablenode=child
            if not variabledict.__contains__(variable):
                variabledict[variable]=[variablenode.id]
            else:
                variabledict[variable].append(variablenode.id)      
        for child in node.children:
            getvariables(child,vocabdict,edgetype,variabledict)
    getvariables(node,vocabdict,edgetype,variabledict)
    #print(variabledict)
    for v in variabledict.keys():
        for i in range(len(variabledict[v])-1):
                src.append(variabledict[v][i])
                tgt.append(variabledict[v][i+1])
                edgetype.append([edges['Nextuse']])
                src.append(variabledict[v][i+1])
                tgt.append(variabledict[v][i])
                edgetype.append([edges['Prevuse']])


def getedge_nextsib(node,vocabdict,src,tgt,edgetype):
    token=node.token
    for i in range(len(node.children)-1):
        src.append(node.children[i].id)
        tgt.append(node.children[i+1].id)
        edgetype.append([1])
        src.append(node.children[i+1].id)
        tgt.append(node.children[i].id)
        edgetype.append([edges['Prevsib']])
    for child in node.children:
        getedge_nextsib(child,vocabdict,src,tgt,edgetype)


def createtree(root,node,nodelist,parent=None):
    id = len(nodelist)
    #print(id)
    token, children = get_token(node), get_child(node)
    if id==0:
        root.token=token
        root.data=node
    else:
        newnode=AnyNode(id=id,token=token,data=node,parent=parent, is_statement=False)
    nodelist.append(node)
    for child in children:
        if id==0:
            createtree(root,child, nodelist, parent=root)
        else:
            createtree(root,child, nodelist, parent=newnode)


def get_sequence(node, sequence):
    token, children = get_token(node), get_child(node)
    sequence.append(token)
    #print(len(sequence), token)
    for child in children:
        get_sequence(child, sequence)


def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    return token


def get_child(root):
    #print(root)
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    #print(sub_item)
                    yield sub_item
            elif item:
                #print(item)
                yield item
    return list(expand(children))
