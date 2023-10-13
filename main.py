import os
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfile, asksaveasfile
import numpy as np
import pandas as pd
import ply.lex as lex
import ply.yacc as yacc
from pandastable import Table
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Globals

PROGRAM_NAME = "Lazy Alchemist"
VERSION = "1.0"

DEFAULT_INGREDIENT_CSV_PATH = "default.csv"

RECIPE_FILE_EXTENSION = ('recipe', '*.rec')

DEFAULT_BASE = 0
DEFAULT_MULTIPLIER = 0
DEFAULT_COST_VALUE = 0.0001

COST_STAT = 'Cost'
AW_STAT = 'AW'

TABS = ["Home",
        "Ingredients",
        "Character",
        "Other Settings",
        "Test Ingredient",
        "Make Potion",
        "Auto-Optimization"]

INGREDIENT_TABLE = None

TESTING_STAT = ""
TESTING_INGREDIENT = ""

TESTING_DATA_POINTS = pd.DataFrame({'ratio': [],
                                    'stat': []})

# TODO: Rework this so that these values aren't hardcoded
NON_STATS = ["Ingredient",
             "AW",
             "Cost",
             "Lore"]
STATS = None


# Globals End

# Parsers
class CostLexer(object):
    # List of token names.   This is always required
    tokens = ('NUMBER', 'VARIABLE')

    literals = ['+', '-', '*', '/', '(', ')']

    # A regular expression rule with some action code
    # Note addition of self parameter since we're in a class
    def __init__(self):
        self.lexer = None

    @staticmethod
    def t_NUMBER(t):
        r"""(\d*\.\d+)|(\d+\.)|\d+"""
        t.value = float(t.value)
        return t

    @staticmethod
    def t_VARIABLE(t):
        r"""[a-zA-Z][a-zA-Z0-9]*"""
        return t

    # A string containing ignored characters (spaces and tabs)
    t_ignore = ' \t\n'

    # Error handling rule
    @staticmethod
    def t_error(t):
        print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)

    # Build the lexer
    def build(self, **kwargs):
        self.lexer = lex.lex(module=self, **kwargs)


# Build the lexer
CostLexer().build()

tokens = CostLexer.tokens

precedence = (
    ('left', '+', '-'),
    ('left', '*', '/'),
    ('right', 'UMINUS'),  # Unary minus operator
)


def p_expression_binop(p):
    """expression : expression '+' expression
                  | expression '-' expression
                  | expression '*' expression
                  | expression '/' expression"""

    p[0] = (p[2], p[1], p[3])


def p_expression_group(p):
    """expression : '(' expression ')'"""
    p[0] = p[2]


def p_expression_number(p):
    """expression : NUMBER"""
    p[0] = ('num', p[1])


def p_factor_variable(p):
    """expression : VARIABLE"""
    p[0] = ('var', p[1])


def p_expr_uminus(p):
    """expression : '-' expression %prec UMINUS"""
    p[0] = ('neg', p[2])


# Error rule for syntax errors
def p_error(p):
    print("Syntax error in input!")


# Build the parser
CostParser = yacc.yacc()


# End Parsers

# Functions

# sum(b_i*r_i)*prod(1+m_i*sqrt(r_i))
def stat_value_func(ingredients_data, ingredients_list, stat):
    S = ingredients_data.loc[ingredients_list][stat].astype('str')

    x = []
    y = []
    for s in S:
        if s == "nan":
            x.append(DEFAULT_BASE)
            y.append(DEFAULT_MULTIPLIER)
        else:
            a, b = s.split(",")
            x.append(float(a))
            y.append(float(b))

    X = np.array(x, float)
    Y = np.array(y, float)

    def calculate(ratios):
        rs = np.array(list(ratios), dtype=float)
        result = np.dot(rs, X) * np.prod(1 + Y * np.sqrt(rs))
        if 0 < result:
            return result
        else:
            return 0

    return calculate


# TODO: Fix corner case where it is optimal to put none of a base in
def make_calc_partial_recipe(ingredients_data, ingredients_list, units):
    a = (10 * units + 1)
    b = ingredients_data.loc[ingredients_list][AW_STAT]
    b[np.isnan(b)] = DEFAULT_COST_VALUE

    def calc_partial_recipe(x):
        return np.array([a * (r / np.dot(x, b)) for r in x])

    return calc_partial_recipe


def cost_value_func(ingredients_data, ingredients_list, calc_partial_recipe):
    X = np.array(ingredients_data.loc[ingredients_list][COST_STAT],
                 dtype=float)

    X[np.isnan(X)] = DEFAULT_COST_VALUE

    def calculate(ratios):
        rs = np.array(list(ratios), dtype=float)
        return np.dot(X, calc_partial_recipe(rs))

    return calculate


def construct_cost_func(ingredients_data, ingredients_list, ast, calc_partial_recipe):
    if ast[0] == '+':
        return lambda x: construct_cost_func(ingredients_data, ingredients_list, ast[1], calc_partial_recipe)(x) + \
                         construct_cost_func(ingredients_data, ingredients_list, ast[2], calc_partial_recipe)(x)
    elif ast[0] == '-':
        return lambda x: construct_cost_func(ingredients_data, ingredients_list, ast[1], calc_partial_recipe)(x) - \
                         construct_cost_func(ingredients_data, ingredients_list, ast[2], calc_partial_recipe)(x)
    elif ast[0] == '*':
        return lambda x: construct_cost_func(ingredients_data, ingredients_list, ast[1], calc_partial_recipe)(x) * \
                         construct_cost_func(ingredients_data, ingredients_list, ast[2], calc_partial_recipe)(x)
    elif ast[0] == '/':
        return lambda x: construct_cost_func(ingredients_data, ingredients_list, ast[1], calc_partial_recipe)(x) / \
                         construct_cost_func(ingredients_data, ingredients_list, ast[2], calc_partial_recipe)(x)
    elif ast[0] == 'neg':
        return lambda x: -construct_cost_func(ingredients_data, ingredients_list, ast[1], calc_partial_recipe)(x)
    elif ast[0] == 'var':
        if ast[1] == 'Cost':
            return cost_value_func(ingredients_data, ingredients_list, calc_partial_recipe)
        else:
            return stat_value_func(ingredients_data, ingredients_list, ast[1])
    elif ast[0] == 'num':
        return lambda _: ast[1]
    else:
        return None


def calc_stat(ingredients_data, ingredients, stat):
    v = np.array(list(ingredients.values()), dtype=int)
    return stat_value_func(ingredients_data, ingredients.keys(), stat)(np.divide(v, np.sum(v)))


# (pu, effects, cost)
def predict(ingredients_data, recipe):
    return (sum([ingredients_data.loc[e[0]][AW_STAT] * e[1] for e in recipe.items()]) - 1) // 10, \
        {stat: calc_stat(ingredients_data, recipe, stat) for stat in STATS}, \
        sum([round(ingredients_data.loc[e[0]][COST_STAT] * e[1], 4)
             if not np.isnan(ingredients_data.loc[e[0]][COST_STAT])
             else round(DEFAULT_COST_VALUE * e[1], 4)
             for e in recipe.items()])


def open_ingredients_file(textvar):
    def open_file():
        file = askopenfile(mode='r', filetypes=[('csv', '*.csv')])
        if file is not None:
            global INGREDIENT_TABLE
            global STATS

            # noinspection PyUnresolvedReferences
            INGREDIENT_TABLE.model.df = pd.read_csv(file)

            STATS = [c for c in INGREDIENT_TABLE.model.df.columns if c not in NON_STATS]

            # noinspection PyUnresolvedReferences
            INGREDIENT_TABLE.redraw()

            textvar.set(os.path.basename(file.name))

    return open_file


def save_ingredients_file(textvar):
    def save_file():
        files = [('csv', '*.csv'),
                 ('All Files', '*.*')]
        file = asksaveasfile(filetypes=files)
        if file is not None:
            global INGREDIENT_TABLE

            # noinspection PyUnresolvedReferences
            INGREDIENT_TABLE.model.df.to_csv(file, index=False)

            textvar.set(os.path.basename(file.name))

    return save_file


# TODO: Prevent the function from using ingredients not in the dataframe
# TODO: Fix bug where a potion with no AW has -1 units
def make_potion_btn(ingredients_list, potion_recipe, potion_result):
    def make_potion():

        recipe = {}
        t = 0
        for e in ingredients_list:
            if (e[1].get() != '') and e[0].get().isdigit():
                if e[1].get() in recipe:
                    recipe[e[1].get()] += int(e[0].get())
                else:
                    recipe[e[1].get()] = int(e[0].get())
                t += int(e[0].get())

        potion_recipe.set('\n'.join(map(lambda e: f"{e[1]} ({round((e[1] / t) * 100, 3)}%) {e[0]}", recipe.items())))

        # noinspection PyUnresolvedReferences
        ingredients_data = INGREDIENT_TABLE.model.df.set_index("Ingredient", inplace=False)

        result = predict(ingredients_data, recipe)

        potion_result.set(f"PU: {result[0]}\n" +
                          '\n'.join(map(lambda e: f"{e[0]}: {round(e[1], 3)}", result[1].items())) +
                          f"\nCost: {result[2]}")

    return make_potion


def optimize_btn(ingredients_list, make_potion, cost_function, units):
    def optimize_potion():

        recipe = {}
        for e in ingredients_list:
            if e[1].get() != '':
                if e[0].get().isdigit():
                    n = int(e[0].get())
                else:
                    n = 0
                if e[1].get() in recipe:
                    recipe[e[1].get()] += n
                else:
                    recipe[e[1].get()] = n

        # noinspection PyUnresolvedReferences
        ingredients_data = INGREDIENT_TABLE.model.df.set_index("Ingredient", inplace=False)
        u = int(units.get())
        calc_partial_recipe = make_calc_partial_recipe(ingredients_data, recipe.keys(), u)

        # TODO: Figure out how to make it allow discontinuities (for rounding)
        obj = lambda x: -construct_cost_func(ingredients_data,
                                             recipe.keys(),
                                             CostParser.parse(cost_function.get()),
                                             calc_partial_recipe)(x)

        c = (lambda x: np.sum(x) - 1)

        vals = np.array(list(recipe.values()), dtype=int)
        vals[vals <= 0] = 1
        x0 = np.divide(vals, np.sum(vals))

        b = (0, 1)
        bnds = tuple(b for _ in range(len(recipe.keys())))

        # TODO: Make the optimization go to completion (starting point has too much influence... Likely want to swap
        #       out the optimization algorithm)
        sol = minimize(obj, x0, bounds=bnds, constraints={'type': 'eq', 'fun': c})

        # TODO: This will likely break when 1 < AW. Might need to fix at some point
        T = 10 * u + 1
        A = sol.x
        B = calc_partial_recipe(A)
        C = (np.rint(B)).astype(int)
        t = np.dot(C, ingredients_data.loc[recipe.keys()][AW_STAT])
        while t != T:
            if t < T:
                C[np.ma.argmin(np.ma.MaskedArray(C - B, ingredients_data.loc[recipe.keys()][AW_STAT] <= 0))] += 1
                t = np.dot(C, ingredients_data.loc[recipe.keys()][AW_STAT])
            else:
                C[np.ma.argmax(np.ma.MaskedArray(C - B, ingredients_data.loc[recipe.keys()][AW_STAT] <= 0))] -= 1
                t = np.dot(C, ingredients_data.loc[recipe.keys()][AW_STAT])

        for j in range(len(C)):
            ingredients_list[j][0].delete(0, tk.END)
            ingredients_list[j][0].insert(0, C[j])

        make_potion()

    return optimize_potion


def clear_btn(ingredients_list):
    def clear():
        for e in ingredients_list:
            e[0].delete(0, tk.END)
            e[1].set('')

    return clear


def load_recipe_btn(ingredients_list, clear):
    def load_recipe():
        file = askopenfile(mode='r', filetypes=[RECIPE_FILE_EXTENSION])
        if file is not None:
            clear()
            j = 0
            for line in file.readlines():
                e0, e1 = line.split(':')
                ingredients_list[j][0].insert(0, e0.strip())
                ingredients_list[j][1].set(e1.strip())
                j += 1

    return load_recipe


def save_recipe_btn(ingredients_list):
    def save_recipe():
        files = [RECIPE_FILE_EXTENSION,
                 ('All Files', '*.*')]
        file = asksaveasfile(defaultextension=RECIPE_FILE_EXTENSION[1], filetypes=files)
        if file is not None:
            for e in ingredients_list:
                if (e[1].get() != '') and e[0].get().isdigit():
                    file.write(f"{e[0].get()}:{e[1].get()}\n")

    return save_recipe


def test_curve(r, b0, m0, b1, m1):
    return (b0 * r + b1 * (1 - r)) * (1 + m0 * np.sqrt(r)) * (1 + m1 * np.sqrt(1 - r))


def calc_b_m(f):
    popt, _ = curve_fit(f=f,
                        xdata=TESTING_DATA_POINTS['ratio'],
                        ydata=TESTING_DATA_POINTS['stat'])

    return popt


def update_datapoints(ax, lbs):
    ax.clear()
    ax.scatter(TESTING_DATA_POINTS['ratio'], TESTING_DATA_POINTS['stat'])

    try:
        if TESTING_INGREDIENT == "" or TESTING_STAT == "":
            if 3 < len(TESTING_DATA_POINTS):
                f = test_curve
                b0, m0, b1, m1 = calc_b_m(f)
                b0 = np.round(b0, 6)
                m0 = np.round(m0, 6)
                b1 = np.round(b1, 6)
                m1 = np.round(m1, 6)

                lbs['b0'].set(b0)
                lbs['m0'].set(m0)
                lbs['b1'].set(b1)
                lbs['m1'].set(m1)

                ax.plot(np.linspace(0, 1),
                        test_curve(np.linspace(0, 1), b0, m0, b1, m1),
                        '--',
                        color='red',
                        label="predicted")
                ax.legend()
        else:
            if 1 < len(TESTING_DATA_POINTS):
                ingredients_data = INGREDIENT_TABLE.model.df.set_index("Ingredient", inplace=False)

                s = str(ingredients_data.loc[TESTING_INGREDIENT][TESTING_STAT])

                if s == "nan":
                    b0 = DEFAULT_BASE
                    m0 = DEFAULT_MULTIPLIER
                else:
                    a, b = s.split(",")
                    b0 = float(a)
                    m0 = float(b)

                f = lambda r, b1, m1: test_curve(r, b0, m0, b1, m1)
                b1, m1 = calc_b_m(f)
                b1 = np.round(b1, 6)
                m1 = np.round(m1, 6)

                lbs['b0'].set(b0)
                lbs['m0'].set(m0)
                lbs['b1'].set(b1)
                lbs['m1'].set(m1)

                ax.plot(np.linspace(0, 1),
                        test_curve(np.linspace(0, 1), b0, m0, b1, m1),
                        '--',
                        color='red',
                        label="predicted")
                ax.legend()
    except:
        pass

    ax.figure.canvas.draw()


# TODO: Potentially prevent duplicate points
def add_datapoint_btn(ax, lbs, r, v):
    def add_datapoint():
        TESTING_DATA_POINTS.loc[len(TESTING_DATA_POINTS)] = {'ratio': float(r.get()), 'stat': float(v.get())}
        update_datapoints(ax, lbs)

    return add_datapoint


def set_testing_ingredient_btn(ax, lbs, ing):
    def set_testing_ingredient():
        global TESTING_INGREDIENT
        TESTING_INGREDIENT = ing.get()
        update_datapoints(ax, lbs)

    return set_testing_ingredient


def set_testing_stat_btn(ax, lbs, stat):
    def set_testing_stat():
        global TESTING_STAT
        TESTING_STAT = stat.get()
        update_datapoints(ax, lbs)

    return set_testing_stat


def clear_datapoints_btn(ax):
    def clear_datapoints():
        global TESTING_DATA_POINTS
        TESTING_DATA_POINTS = pd.DataFrame({'ratio': [],
                                            'stat': []})
        ax.clear()
        ax.figure.canvas.draw()

    return clear_datapoints


def make_ingredients_frame(frame):
    global INGREDIENT_TABLE
    global STATS

    frame.rowconfigure(1, weight=1)
    frame.columnconfigure(0, weight=1)

    frm_top = ttk.Frame(frame, relief=tk.RAISED)

    frm_bot = ttk.Frame(frame)

    INGREDIENT_TABLE = Table(frm_bot, showtoolbar=False, showstatusbar=False)
    INGREDIENT_TABLE.model.df = pd.read_csv(DEFAULT_INGREDIENT_CSV_PATH)

    STATS = [c for c in INGREDIENT_TABLE.model.df.columns if c not in NON_STATS]

    ingredient_file_name = tk.StringVar()
    ingredient_file_name.set(DEFAULT_INGREDIENT_CSV_PATH)

    ttk.Label(frm_top, text="File:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(frm_top, textvariable=ingredient_file_name).grid(row=0, column=1, sticky="w", padx=5, pady=5)
    ttk.Button(frm_top, text="Open", command=open_ingredients_file(ingredient_file_name)) \
        .grid(row=0, column=2, sticky="w", padx=5, pady=5)
    ttk.Button(frm_top, text="Save As...", command=save_ingredients_file(ingredient_file_name)) \
        .grid(row=0, column=3, sticky="w", padx=5, pady=5)

    INGREDIENT_TABLE.show()

    frm_top.grid(row=0, column=0, sticky="ew")
    frm_bot.grid(row=1, column=0, sticky="nsew")


def make_testing_frame(frame):
    frame.rowconfigure(4, weight=1)
    frame.columnconfigure(0, weight=1)

    frm0 = ttk.Frame(frame)

    frm0.rowconfigure(0, weight=1)
    frm0.columnconfigure(0, weight=1)

    figure = plt.Figure(figsize=(5, 3), dpi=100)
    ax = figure.add_subplot(111)
    chart_type = FigureCanvasTkAgg(figure, frm0)
    chart_type.get_tk_widget().grid(row=0, column=0, pady=10)

    frm1 = ttk.Frame(frame)

    frm1.rowconfigure(1, weight=1)
    frm1.columnconfigure(0, weight=1)

    res0 = ttk.Frame(frm1)

    res0.rowconfigure(0, weight=1)
    res0.columnconfigure(3, weight=1)

    lbs = {'b0': tk.DoubleVar(),
           'm0': tk.DoubleVar(),
           'b1': tk.DoubleVar(),
           'm1': tk.DoubleVar()}

    ttk.Label(res0, text="b0:").grid(row=0, column=0, padx=5)
    ttk.Label(res0, textvariable=lbs['b0'], width=10).grid(row=0, column=1, padx=5)
    ttk.Label(res0, text="m0:").grid(row=0, column=2, padx=5)
    ttk.Label(res0, textvariable=lbs['m0'], width=10).grid(row=0, column=3, padx=5)

    res1 = ttk.Frame(frm1)

    res1.rowconfigure(0, weight=1)
    res1.columnconfigure(3, weight=1)

    ttk.Label(res1, text="b1:").grid(row=0, column=0, padx=5)
    ttk.Label(res1, textvariable=lbs['b1'], width=10).grid(row=0, column=1, padx=5)
    ttk.Label(res1, text="m1:").grid(row=0, column=2, padx=5)
    ttk.Label(res1, textvariable=lbs['m1'], width=10).grid(row=0, column=3, padx=5)

    res0.grid(row=0, column=0)
    res1.grid(row=1, column=0)

    frm2 = ttk.Frame(frame)

    frm2.rowconfigure(1, weight=1)
    frm2.columnconfigure(2, weight=1)

    ttk.Label(frm2, text="Testing Ingredient:").grid(row=0, column=0, padx=5, sticky="w")
    ingredient = tk.StringVar()
    # noinspection PyUnresolvedReferences
    ttk.Combobox(frm2,
                 width=30,
                 textvariable=ingredient,
                 values=list(INGREDIENT_TABLE.model.df["Ingredient"])).grid(row=0, column=1, padx=5, sticky="w")
    ttk.Button(frm2,
               text="Set Ingredient",
               command=set_testing_ingredient_btn(ax, lbs, ingredient)).grid(row=0, column=2, padx=5, sticky="w")

    ttk.Label(frm2, text="Stat:").grid(row=1, column=0, padx=5, sticky="w")
    stat = tk.StringVar()
    ttk.Combobox(frm2,
                 width=10,
                 textvariable=stat,
                 values=list(STATS)).grid(row=1, column=1, padx=5, sticky="w")
    ttk.Button(frm2,
               text="Set Stat",
               command=set_testing_stat_btn(ax, lbs, stat)).grid(row=1, column=2, padx=5, sticky="w")

    frm3 = ttk.Frame(frame)

    frm3.rowconfigure(0, weight=1)
    frm3.columnconfigure(4, weight=1)

    ttk.Label(frm3, text="Ratio:").grid(row=0, column=0, padx=5)
    ratio = ttk.Entry(frm3, width=10)
    ratio.grid(row=0, column=1, padx=5)
    ttk.Label(frm3, text="Value:").grid(row=0, column=2, padx=5)
    value = ttk.Entry(frm3, width=10)
    value.grid(row=0, column=3, padx=5)
    ttk.Button(frm3, text="Add", command=add_datapoint_btn(ax, lbs, ratio, value)).grid(row=0, column=4, padx=5)

    frm4 = ttk.Frame(frame)

    frm4.rowconfigure(0, weight=1)
    frm4.columnconfigure(0, weight=1)

    ttk.Button(frm4, text="Clear", command=clear_datapoints_btn(ax)).grid(row=0, column=0)

    frm0.grid(row=0, column=0, pady=10)
    frm1.grid(row=1, column=0, pady=10)
    frm2.grid(row=2, column=0, pady=10)
    frm3.grid(row=3, column=0, pady=10)
    frm4.grid(row=4, column=0, pady=10, sticky="n")


# TODO: This frame's elements need to be updated with the dataframe
def make_make_potion_frame(frame):
    frame.rowconfigure(1, weight=1)
    frame.columnconfigure(1, weight=1)

    frm_tl = ttk.Frame(frame)

    frm_tl.rowconfigure(15, weight=1)
    frm_tl.columnconfigure(1, weight=1)

    make_potion_ingredients = [(ttk.Entry(frm_tl, width=5),
                                tk.StringVar()) for _ in range(16)]

    for j in range(16):
        make_potion_ingredients[j][0].grid(row=j, column=0, padx=1, pady=1)
        # noinspection PyUnresolvedReferences
        ttk.Combobox(frm_tl,
                     width=30,
                     textvariable=make_potion_ingredients[j][1],
                     values=list(INGREDIENT_TABLE.model.df["Ingredient"])).grid(row=j, column=1, padx=10, pady=4)

    frm_tr = ttk.Frame(frame)

    frm_tr.rowconfigure(0, weight=1)
    frm_tr.columnconfigure(0, weight=1)

    frm_pot_result = ttk.Frame(frm_tr)

    frm_pot_result.rowconfigure(1, weight=1)
    frm_pot_result.columnconfigure(0, weight=1)

    frm_pot_result_top = ttk.Frame(frm_pot_result)

    potion_recipe = tk.StringVar()
    potion_recipe.set("")

    ttk.Label(frm_pot_result_top, text="Potion Recipe").grid(row=0, column=0, padx=10, pady=10)
    ttk.Label(frm_pot_result_top, textvariable=potion_recipe).grid(row=1, column=0, padx=10, pady=10)

    frm_pot_result_bot = ttk.Frame(frm_pot_result)

    potion_result = tk.StringVar()
    potion_result.set("")

    ttk.Label(frm_pot_result_bot, text="Potion Result").grid(row=0, column=0, padx=10, pady=10)
    ttk.Label(frm_pot_result_bot, textvariable=potion_result).grid(row=1, column=0, padx=10, pady=10)

    frm_pot_result_top.grid(row=0, column=0)
    frm_pot_result_bot.grid(row=1, column=0)

    frm_pot_result.grid(row=0, column=0)

    frm_bl = ttk.Frame(frame)

    frm_bl.rowconfigure(1, weight=1)
    frm_bl.columnconfigure(1, weight=1)

    make_potion = make_potion_btn(make_potion_ingredients, potion_recipe, potion_result)
    clear = clear_btn(make_potion_ingredients)

    ttk.Button(frm_bl, text="Brew Potion",
               command=make_potion).grid(row=0, column=0)

    ttk.Button(frm_bl, text="Clear Ingredients",
               command=clear).grid(row=0, column=1)

    ttk.Button(frm_bl, text="Load Recipe",
               command=load_recipe_btn(make_potion_ingredients, clear)).grid(row=1, column=0)

    ttk.Button(frm_bl, text="Save Recipe",
               command=save_recipe_btn(make_potion_ingredients)).grid(row=1, column=1)

    frm_br = ttk.Frame(frame)

    frm_br.rowconfigure(2, weight=1)
    frm_br.columnconfigure(1, weight=1)

    ttk.Label(frm_br, text="Cost Function:").grid(row=0, column=0, padx=5, pady=5)
    ttk.Label(frm_br, text="Potion Units:").grid(row=1, column=0, padx=5, pady=5)
    cost_function = ttk.Entry(frm_br, width=28)
    units = ttk.Entry(frm_br, width=10)
    cost_function.grid(row=0, column=1, padx=5, pady=5)
    units.grid(row=1, column=1, padx=5, pady=5)

    ttk.Button(frm_br, text="Optimize",
               command=optimize_btn(make_potion_ingredients,
                                    make_potion,
                                    cost_function,
                                    units)) \
        .grid(row=2, column=0, columnspan=2)

    frm_tl.grid(row=0, column=0, sticky="ns", padx=5, pady=10)
    frm_tr.grid(row=0, column=1, sticky="nsew")
    frm_bl.grid(row=1, column=0, sticky="ns")
    frm_br.grid(row=1, column=1, sticky="nsew")


def change_frame_to(name):
    def btn_change_frame():
        global currentFrame

        if name != currentFrame:
            frames[name].grid(row=0, column=1, sticky="nsew")
            menu_btns[name].state(['pressed'])
            frames[currentFrame].grid_forget()
            menu_btns[currentFrame].state(['!pressed'])
            currentFrame = name

    return btn_change_frame


# Functions End

# GUI

if __name__ == "__main__":

    window = tk.Tk()
    style = ttk.Style()
    window.resizable(width=False, height=False)

    window.title(f"{PROGRAM_NAME} v{VERSION}")

    window.rowconfigure(0, minsize=600, weight=1)
    window.columnconfigure(1, minsize=600, weight=1)

    frm_buttons = ttk.Frame(window, relief=tk.RAISED)

    frames = {x: ttk.Frame(window, relief=tk.RAISED) for x in TABS}

    menu_btns = {x: ttk.Button(frm_buttons, text=x, command=change_frame_to(x)) for x in TABS}

    for i in range(len(TABS)):
        menu_btns[TABS[i]].grid(row=i, column=0, sticky="ew", padx=5, pady=5)

    frm_buttons.grid(row=0, column=0, sticky="ns")

    # Home Frame

    ttk.Label(frames["Home"], text="Made By: LazySpiky").grid(row=0, column=0, padx=40)

    # Ingredients Frame

    make_ingredients_frame(frames["Ingredients"])

    # Character Frame

    ttk.Label(frames["Character"], text="Character Frame").grid(row=0, column=0, padx=40)

    # Other Settings Frame

    ttk.Label(frames["Other Settings"], text="Other Settings Frame").grid(row=0, column=0, padx=40)

    # Test Ingredient Frame

    make_testing_frame(frames["Test Ingredient"])

    # Make Potion Frame

    make_make_potion_frame(frames["Make Potion"])

    # Auto-Optimization Frame

    ttk.Label(frames["Auto-Optimization"], text="Auto-Optimization Frame").grid(row=0, column=0, padx=40)

    # Launch Window

    currentFrame = TABS[0]
    frames[currentFrame].grid(row=0, column=1, sticky="nsew")
    window.mainloop()

# GUI End
