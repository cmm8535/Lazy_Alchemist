
# parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = "left+-left*/rightUMINUSNUMBER VARIABLEexpression : expression '+' expression\n                  | expression '-' expression\n                  | expression '*' expression\n                  | expression '/' expressionexpression : '(' expression ')'expression : NUMBERexpression : VARIABLEexpression : '-' expression %prec UMINUS"
    
_lr_action_items = {'(':([0,2,3,6,7,8,9,],[3,3,3,3,3,3,3,]),'NUMBER':([0,2,3,6,7,8,9,],[4,4,4,4,4,4,4,]),'VARIABLE':([0,2,3,6,7,8,9,],[5,5,5,5,5,5,5,]),'-':([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,],[2,7,2,2,-6,-7,2,2,2,2,-8,7,-1,-2,-3,-4,-5,]),'$end':([1,4,5,10,12,13,14,15,16,],[0,-6,-7,-8,-1,-2,-3,-4,-5,]),'+':([1,4,5,10,11,12,13,14,15,16,],[6,-6,-7,-8,6,-1,-2,-3,-4,-5,]),'*':([1,4,5,10,11,12,13,14,15,16,],[8,-6,-7,-8,8,8,8,-3,-4,-5,]),'/':([1,4,5,10,11,12,13,14,15,16,],[9,-6,-7,-8,9,9,9,-3,-4,-5,]),')':([4,5,10,11,12,13,14,15,16,],[-6,-7,-8,16,-1,-2,-3,-4,-5,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'expression':([0,2,3,6,7,8,9,],[1,10,11,12,13,14,15,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> expression","S'",1,None,None,None),
  ('expression -> expression + expression','expression',3,'p_expression_binop','CostParser.py',16),
  ('expression -> expression - expression','expression',3,'p_expression_binop','CostParser.py',17),
  ('expression -> expression * expression','expression',3,'p_expression_binop','CostParser.py',18),
  ('expression -> expression / expression','expression',3,'p_expression_binop','CostParser.py',19),
  ('expression -> ( expression )','expression',3,'p_expression_group','CostParser.py',25),
  ('expression -> NUMBER','expression',1,'p_expression_number','CostParser.py',30),
  ('expression -> VARIABLE','expression',1,'p_factor_variable','CostParser.py',35),
  ('expression -> - expression','expression',2,'p_expr_uminus','CostParser.py',40),
]
