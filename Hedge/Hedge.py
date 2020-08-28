###########
# IMPORTS #
###########

from dependancy_strings_with_arrows import *

import string 
import os
import math

#############
# CONSTANTS #
#############

DIGITS = "0123456789"
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

##########
# ERRORS #
##########

class Error:
	def __init__(self, posStart, posEnd, errorName, details):
		self.posStart	= posStart
		self.posEnd		= posEnd
		self.errorName  = errorName
		self.details	= details

	def asString(self):
		result  = f'{self.errorName}: {self.details}\n'
		result += f'File {self.posStart.fName}, line {self.posStart.line + 1}'
		result += '\n\n' + string_with_arrows(self.posStart.fText, self.posStart, self.posEnd)
		return result

class IllegalCharError(Error):
	def __init__(self, posStart, posEnd, details):
		super().__init__(posStart, posEnd, 'Illegal Character', details)

class ExpectedCharError(Error):
	def __init__(self, posStart, posEnd, details):
		super().__init__(posStart, posEnd, 'Expected Character', details)

class InvalidSyntaxError(Error):
	def __init__(self, posStart, posEnd, details):
		super().__init__(posStart, posEnd, 'Invalid Syntax', details)

class RTError(Error):
	def __init__(self, posStart, posEnd, details, context):
		super().__init__(posStart, posEnd, 'Runtime Error', details)
		self.context = context

	def asString(self):
		result = self.generateTraceback()
		result += f'{self.errorName}: {self.details}'
		result += '\n\n' + string_with_arrows(self.posStart.fText, self.posStart, self.posEnd)
		return result

	def generateTraceback(self):
		result   = ''
		position = self.posStart
		contxt   = self.context

		while contxt:
			result   = f' File {position.fName}, line {str(position.line + 1)}, in {contxt.displayName}\n' + result
			position = contxt.parentEntryPos
			contxt   = contxt.parent

		return 'Traceback (most recent call last):\n' + result

############
# POSITION #
############

class Position:
	def __init__(self, index, line, column, fName, fText):
		self.index	= index
		self.line	= line
		self.column = column
		self.fName  = fName
		self.fText  = fText

	def advance(self, currentChar = None):
		self.index	+= 1
		self.column += 1

		if currentChar == '\n':
			self.line	+= 1
			self.column  = 0

		return self

	def copy(self):
		return Position(self.index, self.line, self.column, self.fName, self.fText)

##########
# TOKENS #
##########

TT_INT					= 'INT'
TT_FLOAT				= 'FLOAT'
TT_STRING				= 'STRING'
TT_IDENTIFIER			= 'IDENTIFIER'
TT_KEYWORD				= 'KEYWORD'
TT_PLUS					= 'PLUS'			 # +
TT_MINUS				= 'MINUS'			 # -
TT_MULTIPLY				= 'MULTIPLY'		 # *
TT_DIVIDE				= 'DIVIDE'			 # /
TT_POWER				= 'POWER'			 # ^
TT_EQUAL				= 'EQUAL'			 # =
TT_LPAREN				= 'LPAREN'			 # )
TT_RPAREN				= 'RPAREN'			 # (
TT_LBRACKET				= 'LBRACKET'		 # [
TT_RBRACKET				= 'RBRACKET'		 # ]
TT_EQUALEQUAL			= 'EQUALEQUAL'		 # ==
TT_NOTEQUAL				= 'NOTEQUAL'		 # !=
TT_LESSTHAN				= 'LESSTHAN'		 # <
TT_GREATERTHAN			= 'GREATERTHAN'		 # >
TT_LESSTHANEQUAL		= 'LESSTHANEQUAL'	 # <=
TT_GREATERTHANEQUAL		= 'GREATERTHANEQUAL' # >=
TT_COMA					= 'COMA'			 # ,
TT_ARROW				= 'ARROW'			 # ->
TT_NEWLINE				= 'NEWLINE'			 # ; or \n
TT_EOF					= 'EOF'				 # End of file

KEYWORDS = [
	'var',
	'and',
	'or',
	'not',
	'if',
	'elif',
	'else',
	'for',
	'to',
	'step',
	'while',
	'func',
	'then',
	'end',
	'return',
	'continue',
	'break'
]

class Token:
	def __init__(self, type_, value = None, posStart = None, posEnd = None):
		self.type = type_
		self.value = value

		if posStart:
			self.posStart = posStart.copy()
			self.posEnd = posStart.copy()
			self.posEnd.advance()

		if posEnd:
			self.posEnd = posEnd.copy()

	def matches(self, type_, value):
		return self.type == type_ and self.value == value

	def __repr__(self):
		if self.value:
			return f'{self.type}:{self.value}'
		return f'{self.type}'

#########
# LEXER #
#########

class Lexer:
	def __init__(self, fName, text):
		self.fName = fName
		self.text = text
		self.pos = Position(-1, 0, -1, fName, text)
		self.currentChar = None
		self.advance()

	def advance(self):
		self.pos.advance(self.currentChar)
		self.currentChar = self.text[self.pos.index] if self.pos.index < len(self.text) else None

	def makeTokens(self):
		tokens = []

		while self.currentChar != None:
			if self.currentChar in ' \t':
				self.advance()
			elif self.currentChar == '#':
				self.skipComment()
			elif self.currentChar in ';\n':
				tokens.append(Token(TT_NEWLINE, posStart = self.pos))
				self.advance()
			elif self.currentChar in DIGITS:
				tokens.append(self.makeNumber())
			elif self.currentChar in LETTERS:
				tokens.append(self.makeIdentifier())
			elif self.currentChar == '"':
				tokens.append(self.makeString())
			elif self.currentChar == '+':
				tokens.append(Token(TT_PLUS, posStart= self.pos))
				self.advance()
			elif self.currentChar == '-':
				tokens.append(self.makeMinusOrArrow())
			elif self.currentChar == '*':
				tokens.append(Token(TT_MULTIPLY, posStart= self.pos))
				self.advance()
			elif self.currentChar == '/':
				tokens.append(Token(TT_DIVIDE, posStart= self.pos))
				self.advance()
			elif self.currentChar == '^':
				tokens.append(Token(TT_POWER, posStart= self.pos))
				self.advance
			elif self.currentChar == '(':
				tokens.append(Token(TT_LPAREN, posStart= self.pos))
				self.advance
			elif self.currentChar == ')':
				tokens.append(Token(TT_RPAREN, posStart= self.pos))
				self.advance
			elif self.currentChar == '[':
				tokens.append(Token(TT_LBRACKET, posStart= self.pos))
				self.advance
			elif self.currentChar == ']':
				tokens.append(Token(TT_RBRACKET, posStart= self.pos))
				self.advance
			elif self.currentChar == '!':
				token, error = self.makeNotEquals()
				if error:
					return [], error
				tokens.append(token)
			elif self.currentChar == '=':
				tokens.append(self.makeEquals())
			elif self.currentChar == '<':
				tokens.append(self.makeLessThan())
			elif self.currentChar == '>':
				tokens.append(self.makeGreaterThan())
			elif self.currentChar == ',':
				tokens.append(Token(TT_COMA, posStart = self.pos))
				self.advance()
			else:
				posStart = self.pos.copy()
				char = self.currentChar
				self.advance()
				return [], IllegalCharError(posStart, self.pos, "'" + char + "'")

		tokens.append(Token(TT_EOF, posStart = self.pos))
		return tokens, None

	def makeNumber(self):
		numStr = ''
		dotCount = 0
		posStart = self.pos.copy()

		while self.currentChar != None and self.currentChar in DIGITS + '.':
			if self.currentChar == '.':
				if dotCount == 1:
					break
				dotCount += 1
			numStr += self.currentChar
			self.advance()

		if dotCount == 0:
			return Token(TT_INT, int(numStr), posStart, self.pos)
		else:
			return Token(TT_FLOAT, float(numStr), posStart, self.pos)

	def makeString(self):
		string = ''
		posStart = self.pos.copy()
		escapeCharacter = False
		self.advance()

		escapeCharacters = {
			'n': '\n',
			't': '\t'
		}

		while self.currentChar != None and (self.currentChar != '"' or escapeCharacter):
			if escapeCharacter:
				string += escapeCharacters.get(self.currentChar, self.currentChar)
			else:
				if self.currentChar == '\\':
					escapeCharacter = True
				else:
					string += self.currentChar
			self.advance()
			escapeCharacter = False

		self.advance()
		return Token(TT_STRING, string, posStart, self.pos)

	def makeIdentifier(self):
		idStr = ''
		posStart = self.pos.copy()

		while self.currentChar != None and self.currentChar in LETTERS_DIGITS + '_':
			idStr += self.currentChar
			self.advance()

		tokType = TT_KEYWORD if idStr in KEYWORDS else TT_IDENTIFIER
		return Token(tokType, idStr, posStart, self.pos)

	def makeMinusOrArrow(self):
		tokType = TT_MINUS
		posStart = self.pos.copy()
		self.advance()

		if self.currentChar == '>':
			self.advance()
			tokType = TT_ARROW

		return Token(tokType, posStart = posStart, posEnd = self.pos)
	
	def makeNotEquals(self):
		posStart = self.pos.copy()
		self.advance()

		if self.currentChar == '=':
			self.advance()
			return Token(TT_NOTEQUAL, posStart = posStart, posEnd = self.pos), None

		self.advance()
		return None, ExpectedCharError(posStart, self.pos, "'=' (after '!'")

	def makeEquals(self):
		tokType = TT_EQUAL
		posStart = self.pos.copy()
		self.advance()

		if self.currentChar == '=':
			self.advance()
			tokType = TT_EQUALEQUAL

		return Token(tokType, posStart = posStart, posEnd = self.pos)

	def makeLessThan(self):
		tokType = TT_LESSTHAN
		posStart = self.pos.copy()
		self.advance()

		if self.currentChar == '=':
			self.advance
			tokType = TT_LESSTHANEQUAL

		return Token(tokType, posStart = posStart, posEnd = self.pos)

	def makeGreaterThan(self):
		tokType = TT_GREATERTHAN
		posStart = self.pos.copy()
		self.advance()

		if self.currentChar == '=':
			self.advance()
			tokType = TT_GREATERTHANEQUAL

		return Token(tokType, posStart = posStart, posEnd = self.pos)

	def skipComment(self):
		self.advance()

		while self.currentChar != '\n':
			self.advance()

		self.advance()

#########
# NODES #
#########

class NumberNode:
	def __init__(self, tok):
		self.tok = tok

		self.posStart = self.tok.posStart
		self.posEnd = self.tok.posEnd

	def __repr__(self):
		return f'{self.tok}'

class StringNode:
	def __init__(self, tok):
		self.tok = tok

		self.posStart = self.tok.posStart
		self.posEnd = self.tok.posEnd

	def __repr__(self):
		return f'{self.tok}'

class ListNode:
	def __init__(self, elementNodes, posStart, posEnd):
		self.elementNodes = elementNodes

		self.posStart = posStart
		self.posEnd = posEnd

class VarAccessNode:
	def __init__(self, varNameTok):
		self.varNameTok = varNameTok

		self.posStart = self.varNameTok.posStart
		self.posEnd = self.varNameTok.posEnd

class VarAssignNode:
	def __init__(self, varNameTok, valueNode):
		self.varNameTok = varNameTok
		self.valueNode =  valueNode

		self.posStart = self.varNameTok.posStart
		self.posEnd = self.valueNode.posEnd

class BinOpNode:
	def __init__(self, leftNode, opTok, rightNode):
		self.leftNode = leftNode
		self.opTok = opTok
		self.rightNode = rightNode

		self.posStart = self.leftNode.posStart
		self.posEnd = self.rightNode.posEnd

	def __repr__(self):
		return f'({self.leftNode}, {self.opTok}, {self.rightNode})'

class UnaryOpNode:
	def __init__(self, opTok, node):
		self.opTok = opTok
		self.node = node

		self.posStart = self.opTok.posStart
		self.posEnd = node.posEnd

	def __repr__(self):
		return f'({self.opTok}, {self.node})'

class IfNode:
	def __init__(self, cases, elseCase):
		self.cases = cases
		self.elseCase = elseCase

		self.posStart = self.cases[0][0].posStart
		self.posEnd = (self.elseCase or self.cases[len(self.cases) - 1])[0].posEnd

class ForNode:
	def __init__(self, varNameTok, startValueNode, endValueNode, stepValueNode, bodyNode, shouldReturnNull):
		self.varNameTok = varNameTok
		self.startValueNode = startValueNode
		self.endValueNode = endValueNode
		self.stepValueNode = stepValueNode
		self.bodyNode = bodyNode
		self.shouldReturnNull = shouldReturnNull

		self.posStart = self.varNameTok.posStart
		self.posEnd = self.bodyNode.posEnd

class WhileNode:
	def __init__(self, conditionNode, bodyNode, shouldReturnNull):
		self.conditionNode = conditionNode
		self.bodyNode = bodyNode
		self.shouldReturnNull = shouldReturnNull

		self.posStart = self.conditionNode.posStart
		self.posEnd = self.bodyNode.posEnd

class FuncDefNode:
	def __init__(self, varNameTok, argNameToks, bodyNode, shouldAutoReturn):
		self.varNameTok = varNameTok
		self.argNameToks = argNameToks
		self.bodyNode = bodyNode
		self.shouldAutoReturn = shouldAutoReturn

		if self.varNameTok:
			self.posStart = self.varNameTok.posStart
		elif len(self.argNameToks) > 0:
			self.posStart = self.argNameToks[0].posStart
		else:
			self.posStart = self.bodyNode.posStart

		self.posEnd = self.bodyNode.posEnd

class CallNode:
	def __init__(self, nodeToCall, argNodes):
		self.nodeToCall = nodeToCall
		self.argNodes = argNodes

		self.posStart = self.nodeToCall.posStart

		if len(self.argNodes) > 0:
			self.posEnd = self.argNodes[len(self.argNodes) - 1].posEnd
		else:
			self.posEnd = self.nodeToCall.posEnd

class ReturnNode:
	def __init__(self, nodeToReturn, posStart, posEnd):
		self.nodeToReturn = nodeToReturn

		self.posStart = posStart
		self.posEnd = posEnd

class ContinueNode:
	def __init__(self, posStart, posEnd):
		self.posStart = posStart
		self.posEnd = posEnd

class BreakNode:
	def __init__(self, posStart, posEnd):
		self.posStart = posStart
		self.posEnd = posEnd

################
# PARSE RESULT #
################

class ParseResult:
	def __init__(self):
		self.error = None
		self.node = None
		self.lastRegisteredAdvanceCount = 0
		self.advanceCount = 0
		self.toReverseCount = 0

	def registerAdvancement(self):
		self.lastRegisteredAdvanceCount = 1
		self.advanceCount += 1

	def register(self, res):
		self.lastRegisteredAdvanceCount = res.advanceCount
		self.advanceCount += res.advanceCount
		if res.error:
			self.error = res.error
		return res.node

	def tryRegister(self, res):
		if res.error:
			self.toReverseCount = res.advanceCount
			return None
		return self.register(res)

	def success(self, node):
		self.node = node

		return self

	def failure(self, error):
		if not self.error or self.lastRegisteredAdvanceCount == 0:
			self.error = error

		return self

##########
# PARSER #
##########

class Parser:
	def __init__(self, tokens):
		self.tokens = tokens
		self.tokIndex = -1
		self.advance()

	def advance(self):
		self.tokIndex += 1
		self.updateCurrentTok()

		return self.currentTok

	def reverse(self, amount = 1):
		self.tokIndex -= amount
		self.updateCurrentTok()

		return self.currentTok

	def updateCurrentTok(self):
		if self.tokIndex >= 0 and self.tokIndex < len(self.tokens):
			self.currentTok = self.tokens[self.tokIndex]

	def parse(self):
		res = self.statements()
		if not res.error and self.currentTok.type != TT_EOF:
			return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, "Token cannot appear after previous tokens"))

		return res

	##############################################################################

	def statements(self):
		res = ParseResult()
		statements = []
		posStart = self.currentTok.posStart.copy()

		while self.currentTok.type == TT_NEWLINE:
			res.registerAdvancement()
			self.advance()

		statement = res.register(self.statement())
		if res.error:
			return res
		statements.append(statement)

		moreStatements = True

		while True:
			newlineCount = 0
			while self.currentTok.type == TT_NEWLINE:
				res.registerAdvancement()
				self.advance()
				newlineCount += 1

			if newlineCount == 0:
				moreStatements = False

			if not moreStatements:
				break
			statement = res.tryRegister(self.statement())
			if not statement:
				self.reverse(res.toReverseCount)
				moreStatements = False
				continue
			statements.append(statement)

		return res.success(ListNode(statements, posStart, self.currentTok.posEnd.copy()))

	def statement(self):
		res = ParseResult()
		posStart = self.currentTok.posStart.copy()

		if self.currentTok.matches(TT_KEYWORD, 'return'):
			res.registerAdvancement()
			self.advance()

			expr = res.tryRegister(self.expr())
			if not expr:
				self.reverse(res.toReverseCount)
			return res.success(ReturnNode(expr, posStart, self.currentTok.posStart.copy()))

		if self.currentTok.matches(TT_KEYWORD, 'continue'):
			res.registerAdvancement()
			self.advance()
			return res.success(ContinueNode(posStart, self.currentTok.posStart.copy()))

		if self.currentTok.matches(TT_KEYWORD, 'break'):
			res.registerAdvancement()
			self.advance()
			return res.success(BreakNode(posStart, self.currentTok.posStart.copy()))

		expr = res.register(self.expr())

		if res.error:
			return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, 
										 "Expected 'return', 'continue', 'break', 'var', 'if', 'for', 'while', 'func', int, float, identifier, '+', '-', '(', '[' or 'not'"))
		return res.success(expr)

	def expr(self):
		res = ParseResult()

		if self.currentTok.matches(TT_KEYWORD, 'var'):
			res.registerAdvancement()
			self.advance()

			if self.currentTok.type != TT_IDENTIFIER:
				return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, "Expected identifier"))

			varName = self.currentTok
			res.registerAdvancement()
			self.advance()

			if self.currentTok.type != TT_EQUAL:
				return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, "Expected '='"))

			res.registerAdvancement()
			self.advance()
			expr = res.register(self.expr())
			if res.error:
				return res
			return res.success(VarAssignNode(varName, expr))

		node = res.register(self.binOp(self.compExpr, ((TT_KEYWORD, 'and'),(TT_KEYWORD, 'or'))))

		if res.error:
			return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, 
										 "Expected 'var', 'if', 'for', 'while', 'func', int, float, identifier, '+', '-', '(', '[' or 'not'"))

		return res.success(node)

	def compExpr(self):
		res = ParseResult()

		if self.currentTok.matches(TT_KEYWORD, 'not'):
			opTok = self.currentTok
			res.registerAdvancement()
			self.advance()

			node = res.register(self.compExpr())
			if res.error:
				return res
			return res.success(UnaryOpNode(opTok, node))

		node = res.register(self.binOp(self.arithExpr, (TT_EQUALEQUAL, TT_NOTEQUAL, TT_LESSTHAN, TT_GREATERTHAN, TT_LESSTHANEQUAL, TT_GREATERTHANEQUAL)))

		if res.error:
			return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd,
										 "Expected 'if', 'for', 'while', 'func', int, float, identifier, '+', '-', '(', '[' or 'not'"))

		return res.success(node)

	def arithExpr(self):
		return self.binOp(self.term, (TT_PLUS, TT_MINUS))

	def term(self):
		return self.binOp(self.factor, (TT_MULTIPLY, TT_DIVIDE))

	def factor(self):
		res = ParseResult()
		tok = self.currentTok

		if tok.type in (TT_PLUS, TT_MINUS):
			res.registerAdvancement()
			self.advance()
			factor = res.register(self.factor())
			if res.error:
				return res
			return res.success(UnaryOpNode(tok, factor))

		return self.power()

	def power(self):
		return self.binOp(self.call, (TT_POWER, ), self.factor)

	def call(self):
		res = ParseResult()
		atom = res.register(self.atom())
		if res.error:
			return res

		if self.currentTok.type == TT_LPAREN:
			res.registerAdvancement()
			self.advance()
			argNodes = []

			if self.currentTok.type == TT_RPAREN:
				res.registerAdvancement()
				self.advance()
			else:
				argNodes.append(res.register(self.expr()))
				if res.error:
					return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, 
										   "Expected ')', 'var', 'if', 'for', 'while', 'func', int, float, identifier, '+', '-', '(', '[' or 'not'"))

				while self.currentTok.type == TT_COMA:
					res.registerAdvancement()
					self.advance()
					
					argNodes.append(res.register(self.expr()))
					if res.error:
						return res

				if self.currentTok.type != TT_RPAREN:
					return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected ',' or ')'"))

				res.registerAdvancement()
				self.advance()
			
			return res.success(CallNode(atom, argNodes))
		return res.success(atom)

	def atom(self):
		res = ParseResult()
		tok = self.currentTok

		if tok.type in (TT_INT, TT_FLOAT):
			res.registerAdvancement()
			self.advance()
			return res.success(NumberNode(tok))
		elif tok.type == TT_STRING:
			res.registerAdvancement()
			self.advance()
			return res.success(StringNode(tok))
		elif tok.type == TT_IDENTIFIER:
			res.registerAdvancement()
			self.advance()
			return res.success(VarAccessNode(tok))
		elif tok.type == TT_LPAREN:
			res.registerAdvancement()
			self.advance()
			expr = res.register(self.expr())
			if res.error:
				return res
			if self.currentTok.type == TT_RPAREN:
				res.registerAdvancement()
				self.advance()
				return res.success(expr)
			else:
				return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, "Expected ')'"))
		elif tok.type == TT_LBRACKET:
			listExpr = res.register(self.listExpr())
			if res.error:
				return res
			return res.success(listExpr)
		elif tok.matches(TT_KEYWORD, 'if'):
			ifExpr = res.register(self.ifExpr())
			if res.error:
				return res
			return res.success(ifExpr)
		elif tok.matches(TT_KEYWORD, 'for'):
			forExpr = res.register(self.forExpr())
			if res.error:
				return res
			return res.success(forExpr)
		elif tok.matches(TT_KEYWORD, 'while'):
			whileExpr = res.register(self.whileExpr())
			if res.error:
				return res
			return res.success(whileExpr)
		elif tok.matches(TT_KEYWORD, 'func'):
			funcDef = res.register(self.funcDef())
			if res.error:
				return res
			return res.success(funcDef)

		return res.failure(InvalidSyntaxError(tok.posStart, tok.posEnd, "Expected int, float, identifier, '+', '-', '(', '[', 'if', 'for', 'while', 'func'"))

	def listExpr(self):
		res = ParseResult()
		elementNodes = []
		posStart = self.currentTok.posStart.copy()

		if self.currentTok.type != TT_LBRACKET:
			return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected '['"))

		res.registerAdvancement()
		self.advance()

		if self.currentTok.type == TT_RBRACKET:
			res.registerAdvancement()
			self.advance()
		else:
			elementNodes.append(res.register(self.expr()))
			if res.error:
				return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, 
										  "Expected ']', 'var', 'if', 'for', 'while', 'func', int, float, identifier, '+', '-', '(', '[' or 'not'"))

			while self.currentTok.type == TT_COMA:
				res.registerAdvancement()
				self.advance()

				elementNodes.append(res.register(self.expr()))
				if res.error:
					return res

			if self.currentTok.type != TT_RBRACKET:
				return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected ',' or ']'"))

			res.registerAdvancement()
			self.advance()

		return res.success(ListNode(elementNodes, posStart, self.currentTok.posEnd.copy()))

	def ifExpr(self):
		res = ParseResult()
		allCases = res.register(self.ifExprCases('if'))
		if res.error:
			return res
		cases, elseCase = allCases
		return res.success(IfNode(cases, elseCase))

	def ifExprB(self):
		return self.ifExprCases('elif')

	def ifExprC(self):
		res = ParseResult()
		elseCase = None

		if self.currentTok.matches(TT_KEYWORD, 'else'):
			res.registerAdvancement()
			self.advance()

			if self.currentTok.type == TT_NEWLINE:
				res.registerAdvancement()
				self.advance()

				statements = res.register(self.statements())
				if res.error:
					return res
				elseCase = (statements, True)

				if self.currentTok.matches(TT_KEYWORD, 'end'):
					res.registerAdvancement()
					self.advance()
				else:
					return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, "Expected 'end'"))
			else:
				expr = res.register(self.statement())
				if res.error:
					return res
				elseCase = (expr, False)

		return res.success(elseCase)

	def ifExprBOrC(self):
		res = ParseResult()
		cases, elseCase = [], None

		if self.currentTok.matches(TT_KEYWORD, 'elif'):
			allCases = res.register(self.ifExprB())
			if res.error:
				return res
			cases, elseCase = allCases
		else:
			elseCase = res.register(self.ifExprC())
			if res.error:
				return res

		return res.success((cases, elseCase))

	def ifExprCases(self, caseKeyword):
		res = ParseResult()
		cases = []
		elseCase = None

		if not self.currentTok.matches(TT_KEYWORD, caseKeyword):
			return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected '{caseKeyword}'"))

		res.registerAdvancement()
		self.advance()

		condition = res.register(self.expr())
		if res.error:
			return res

		if not self.currentTok.matches(TT_KEYWORD, 'then'):
			return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected 'then'"))

		res.registerAdvancement()
		self.advance()

		if self.currentTok.type == TT_NEWLINE:
			res.registerAdvancement()
			self.advance()

			statements = res.register(self.statements())
			if res.error:
				return res
			cases.append((condition, statements, True))

			if self.currentTok.matches(TT_KEYWORD, 'end'):
				res.registerAdvancement()
				self.advance()
			else:
				allCases = res.register(ifExprBOrC())
				if res.error:
					return res
				newCases, elseCase = allCases
				cases.extend(newCases)
		else:
			expr = res.register(self.statement())
			if res.error:
				return res
			cases.append((condition, expr, False))

			allCases = res.register(self.ifExprBOrC())
			if res.error:
				return res
			newCases, elseCase = allCases
			cases.extend(newCases)

		return res.success((cases, elseCase))

	def forExpr(self):
		res = ParseResult()

		if not self.currentTok.matches(TT_KEYWORD, 'for'):
			return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected 'for'"))

		res.registerAdvancement()
		self.advance()

		if self.currentTok.type != TT_IDENTIFIER:
			return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f'Expected identifer'))

		varName = self.currentTok
		res.registerAdvancement()
		self.advance()

		if self.currentTok.type != TT_EQUAL:
			return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected '='"))

		res.registerAdvancement()
		self.advance()

		startValue = res.register(self.expr())
		if res.error:
			return res

		if not self.currentTok.matches(TT_KEYWORD, 'to'):
			return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected 'to'"))

		res.registerAdvancement()
		self.advance()

		endValue = res.register(self.expr())
		if res.error:
			return res

		if self.currentTok.matches(TT_KEYWORD, 'step'):
			res.registerAdvancement()
			self.advance()

			stepValue = res.register(self.expr())
			if res.error:
				return res
		else:
			stepValue = None


		if not self.currentTok.matches(TT_KEYWORD, 'then'):
			return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected 'then"))

		res.registerAdvancement()
		self.advance()

		if self.currentTok.type == TT_NEWLINE:
			res.registerAdvancement()
			self.advance()

			body = res.register(self.statements())
			if res.error:
				return res

			if not self.currentTok.matches(TT_KEYWORD, 'end'):
				return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected 'end'"))

			res.registerAdvancement()
			self.advance()

			return res.success(ForNode(varName, startValue, endValue, stepValue, body, True))

		body = res.register(self.statement())
		if res.error:
			return res

		return res.success(ForNode(varName, startValue, endValue, stepValue, body, False))

	def whileExpr(self):
		res = ParseResult()

		if not self.currentTok.matches(TT_KEYWORD, 'while'):
			return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected 'while'"))
		
		res.registerAdvancement()
		self.advance()

		condition = res.register(self.expr())
		if res.error:
			return res

		if not self.currentTok.matches(TT_KEYWORD, 'then'):
			return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected 'then'"))

		res.registerAdvancement()
		self.advance()

		if self.currentTok.type == TT_NEWLINE:
			res.registerAdvancement()
			self.advance()

			body = res.register(self.statements())
			if res.error:
				return res

			if not self.currentTok.matches(TT_KEYWORD, 'end'):
				return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected 'end'"))

			res.registerAdvancement()
			self.advance()

			return res.success(WhileNode(condition, body, True))

		body = res.register(self.statement())
		if res.error:
			return res

		return res.success(WhileNode(condition, body, False))

	def funcDef(self):
		res = ParseResult()

		if not self.currentTok.matches(TT_KEYWORD, 'func'):
			return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected 'func'"))

		res.registerAdvancement()
		self.advance()

		if self.currentTok.type == TT_IDENTIFIER:
			varNameTok = self.currentTok
			res.registerAdvancement()
			self.advance()

			if self.currentTok.type != TT_LPAREN:
				return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected '('"))
		else:
			varNameTok = None
			if self.currentTok.type != TT_LPAREN:
				return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected '('"))

		res.registerAdvancement()
		self.advance()
		argNameToks = []

		if self.currentTok.type == TT_IDENTIFIER:
			argNameToks.append(self.currentTok)
			res.registerAdvancement()
			self.advance()

			while self.currentTok.type == TT_COMA:
				res.registerAdvancement()
				self.advance()

				if self.currentTok.type != TT_IDENTIFIER:
					return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected identifier"))

				argNameToks.append(self.currentTok)
				res.registerAdvancement()
				self.advance()

			if self.currentTok.type != TT_RPAREN:
				return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected ',' or ')'"))
		else:
			if self.currentTok.type != TT_RPAREN:
				return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected identifier or ')'"))

		res.registerAdvancement()
		self.advance()

		if self.currentTok.type == TT_ARROW:
			res.registerAdvancement()
			self.advance()

			body = res.register(self.expr())
			if res.error:
				return res

			return res.success(FuncDefNode(varNameTok, argNameToks, body, True))

		if self.currentTok.type != TT_NEWLINE:
			return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected '->' or 'newline'"))

		res.registerAdvancement()
		self.advance()

		body = res.register(self.statements())
		if res.error:
			return res

		if not self.currentTok.matches(TT_KEYWORD, 'end'):
			return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected 'end'"))

		res.registerAdvancement()
		self.advance()

		return res.success(FuncDefNode(varNameTok, argNameToks, body, False))

	################################################################

	def binOp(self, funcA, ops, funcB = None):
		if funcB == None:
			funcB = funcA

		res = ParseResult()

		left = res.register(funcA())
		if res.error:
			return res

		while self.currentTok.type in ops or (self.currentTok.type, self.currentTok.value) in ops:
			opTok = self.currentTok
			res.registerAdvancement()
			self.advance()
			right = res.register(funcB())
			if res.error:
				return res
			left = BinOpNode(left, opTok, right)

		return res.success(left)

##################
# RUNTIME RESULT #
##################

class RTResult:
	def __init__(self):
		self.reset()

	def reset(self):
		self.value = None
		self.error = None
		self.funcReturnValue = None
		self.loopShouldContinue = False
		self.loopShouldBreak = False

	def register(self, res):
		self.error = res.error
		self.funcReturnValue = res.funcReturnValue
		self.loopShouldContinue = res.loopShouldContinue
		self.loopShouldBreak = res.loopShouldBreak
		return res.value

	def success(self, value):
		self.reset()
		self.value = value
		return self

	def successReturn(self, value):
		self.reset()
		self.funcReturnValue = value
		return self

	def successContinue(self):
		self.reset()
		self.loopShouldContinue = True
		return self

	def successBreak(self):
		self.reset()
		self.loopShouldBreak = True
		return self

	def failure(self, error):
		self.reset()
		self.error = error
		return self

	# This function will allow you to continue and break outside of the current function
	def shouldReturn(self):
		return (self.error or self.funcReturnValue or self.loopShouldContinue or self.loopShouldBreak)

##########
# VALUES #
##########

class Value:
	def __init__(self):
		self.setPos()
		self.setContext()

	def setPos(self, posStart = None, posEnd = None):
		self.posStart = posStart
		self.posEnd = posEnd
		return self

	def setContext(self, context = None):
		self.context = context
		return self

	def addedTo(self, other):
		return None, self.illegalOperation(other)

	def subtractedBy(self, other):
		return None, self.illegalOperation(other)

	def multipliedBy(self, other):
		return None, self.illegalOperation(other)

	def dividedBy(self, other):
		return None, self.illegalOperation(other)

	def poweredBy(self, other):
		return None, self.illegalOperation(other)

	def getComparisonEqual(self, other):
		return None, self.illegalOperation(other)

	def getComparisonNotEqual(self, other):
		return None, self.illegalOperation(other)
	
	def getComparisonLessThan(self, other):
		return None, self.illegalOperation(other)

	def getComparisonGreaterThan(self, other):
		return None, self.illegalOperation(other)

	def getComparisonLessThanEqual(self, other):
		return None, self.illegalOperation(other)

	def getComparisonGreaterThanEqual(self, other):
		return None, self.illegalOperation(other)

	def andedBy(self, other):
		return None, self.illegalOperation(other)

	def oredBy(self, other):
		return None, self.illegalOperation(other)

	def notted(self):
		return None, self.illegalOperation(other)

	def execute(self, args):
		return RTResult().failure(self.illegalOperation())

	def copy(self):
		raise Exception('No copy method defined')

	def isTrue(self):
		return False
	
	def illegalOperation(self, other = None):
		if not other:
			other = self
		return RTError(self.posStart, other.posEnd, 'Illegal operation', self.context)

class Number(Value):
	def __init__(self, value):
		super().__init__()
		self.value = value

	def addedTo(self, other):
		if isinstance(other, Number):
			return Number(self.value + other.value).setContext(self.context), None
		else:
			return None, Value.illegalOperation(self, other)

	def subtractedBy(self, other):
		if isinstance(other, Number):
			return Number(self.value - other.value).setContext(self.context), None
		else:
			return None, Value.illegalOperation(self, other)

	def multipliedBy(self, other):
		if isinstance(other, Number):
			return Number(self.value * other.value).setContext(self.context), None
		else:
			return None, Value.illegalOperation(self, other)

	def dividedBy(self, other):
		if isinstance(other, Number):
			if other.value == 0:
				return None, RTError(other.posStart, other.posEnd, 'Division by zero', self.context)

			return Number(self.value / other.value).setContext(self.context), None
		else:
			return None, Value.illegalOperation(self, other)

	def poweredBy(self, other):
		if isinstance(other, Number):
			return Number(self.value ** other.value).setContext(self.context), None
		else:
			return None, Value.illegalOperation(self, other)

	def getComparisonEqual(self, other):
		if isinstance(other, Number):
			return Number(int(self.value == other.value)).setContext(self.context), None
		else:
			return None, Value.illegalOperation(self, other)

	def getComparisonNotEqual(self, other):
		if isinstance(other, Number):
			return Number(int(self.value != other.value)).setContext(self.context), None
		else:
			return None, Value.illegalOperation(self, other)

	def getComparisonLessThan(self, other):
		if isinstance(other, Number):
			return Number(int(self.value < other.value)).setContext(self.context), None
		else:
			return None, Value.illegalOperation(self, other)

	def getComparisonGreaterThan(self, other):
		if isinstance(other, Number):
			return Number(int(self.value > other.value)).setContext(self.context), None
		else:
			return None, Value.illegalOperation(self, other)

	def getComparisonLessThanEqual(self, other):
		if isinstance(other, Number):
			return Number(int(self.value <= other.value)).setContext(self.context), None
		else:
			return None, Value.illegalOperation(self, other)

	def getComparisonGreaterThanEqual(self, other):
		if isinstance(other, Number):
			return Number(int(self.value >= other.value)).setContext(self.context), None
		else:
			return None, Value.illegalOperation(self, other)

	def andedBy(self, other):
		if isinstance(other, Number):
			return Number(int(self.value and other.value)).setContext(self.context), None
		else:
			return None, Value.illegalOperation(self, other)

	def oredBy(self, other):
		if isinstance(other, Number):
			return Number(int(self.value or other.value)).setContext(self.context), None
		else:
			return None, Value.illegalOperation(self, other)

	def notted(self):
		return Number(1 if self.value == 0 else 0).setContext(self.context), None

	def copy(self):
		copy = Number(self.value)
		copy.setPos(self.posStart, self.posEnd)
		copy.setContext
		return copy

	def isTrue(self):
		return self.value != 0

	def __str__(self):
		return str(self.value)

	def __repr__(self):
		return str(self.value)

Number.null = Number(0)
Number.false = Number(0)
Number.true = Number(1)
Number.mathPI = Number(math.pi)

class String(Value):
	def __init__(self, value):
		super().__init__()
		self.value = value

	def addedTo(self, other):
		if isinstance(other, String):
			return String(self.value + self.other).setContext(self.context), None
		else:
			return None, Value.illegalOperation(self, other)

	def multipliedBy(self, other):
		if isinstance(other, String):
			return String(self.value * self.other).setContext(self.context), None
		else:
			return None, Value.illegalOperation(self, other)

	def isTrue(self):
		return len(self.value) > 0

	def copy(self):
		copy = String(self.value)
		copy.setPos(self.posStart, self.posEnd)
		copy.setContext(self.context)
		return copy

	def __str__(self):
		return self.value

	def __repr__(self):
		return f'"{self.value}"'

class List(Value):
	def __init__(self, elements):
		super().__init__()
		self.elements = elements

	def addedTo(self, other):
		newList = self.copy()
		newList.elements.append(other)
		return newList, None

	def subtractedBy(self, other):
		if isinstance(other, Number):
			newList = self.copy()
			try:
				newList.elements.pop(other.value)
				return newList, None
			except:
				return None, RTError(other.posStart, other.posEnd, "Element at this index could not be removed from list because index is out of bounds", self.context)
		else:
			return None, Value.illegalOperation(self, other)

	def multipliedBy(self, other):
		if isinstance(other, List):
			newList = self.copy()
			newList.elements.extend(other.elements)
			return newList, None
		else:
			return None, Value.illegalOperation(self, other)

	def dividedBy(self, other):
		if isinstance(self, Number):
			try:
				return self.elements[other.value], None
			except:
				return None, RTError(other.posStart, other.posEnd, "Element at this index could not be retrived from list because index is out of bounds", self.context)
		else:
			return None, Value.illegalOperation(self, other)

	def copy(self):
		copy = List(self.elements)
		copy.setPos(self.posStart, self.posEnd)
		copy.setContext(self.context)
		return copy

	def __str__(self):
		return ", ".join([str(x) for x in self.elements])

	def __repr__(self):
		return f'[{", ".join([repr(x) for x in self.elements])}]'

class BaseFunction(Value):
	def __init__(self, name):
		super().__init__()
		self.name = name or "<anonymous>"

	def generateNewContext(self):
		newContext = Context(self.name, self.context, self.posStart)
		newContext.symbolTable = SymbolTable(newContext.parent.symbolTable)
		return newContext

	def checkArgs(self, argNames, args):
		res = RTResult()

		if len(args) > len(argNames):
			return res.failure(RTError(self.posStart, self.posEnd, f"{len(args) - len(argNames)} too many args passed into {self}", self.context))

		if len(args) < len(argNames):
			return res.failure(RTError(self.posStart, self.posEnd, f"{len(args) - len(argNames)} too few args passed into {self}", self.context))

		return res.success(None)

	def populateArgs(self, argNames, args, executeContext):
		for i in range(len(args)):
			argName = argNames[i]
			argValue = args[i]
			argValue.setContext(executeContext)
			executeContext.symbolTable.set(argName, argValue)

	def checkAndPopulateArgs(self, argNames, args, executeContext):
		res = RTResult()
		res.register(self.checkArgs(argNames, args))
		if res.shouldReturn():
			return res
		self.populateArgs(argNames, args, executeContext)
		return res.success(None)

class Function(BaseFunction):
	def __init__(self, name, bodyNode, argNames, shouldAutoReturn):
		super().__init__(name)
		self.bodyNode = bodyNode
		self.argNames = argNames
		self.shouldAutoReturn = shouldAutoReturn

	def execute(self, args):
		res = RTResult()
		interpreter = Interpreter()
		executeContext = self.generateNewContext()

		res.register(self.checkAndPopulateArgs(self.argNames, args, executeContext))
		if res.shouldReturn():
			return res

		value = res.register(interpreter.visit(self.bodyNode, executeContext))
		if res.shouldReturn() and res.funcReturnValue == None:
			return res

		returnValue = (value if self.shouldAutoReturn else None) or res.funcReturnValue or Number.null
		return res.success(returnValue)

	def copy(self):
		copy = Function(self.name, self.bodyNode, self.argNames, self.shouldAutoReturn)
		copy.setContext(self.context)
		copy.setPos(self.posStart, self.posEnd)
		return copy

	def __repr__(self):
		return f"<function {self.name}>"

class BuiltInFunction(BaseFunction):
	def __init__(self, name):
		super().__init__(name)

	def execute(self, args):
		res = RTResult()
		executeContext = self.generateNewContext()

		methodName = f'execute{self.name}'
		method = getattr(self, methodName, self.noVisitMethod)
		
		res.register(self.checkAndPopulateArgs(method.argNames, args, executeContext))
		if res.shouldReturn():
			return res

		returnValue = res.register(method(executeContext))
		if res.shouldReturn():
			return res
		return res.success(returnValue)

	def noVisitMethod(self, node, context):
		raise Exception(f"No execute{self.name} method defined")

	def copy(self):
		copy = BuiltInFunction(self.name)
		copy.setContext(self.context)
		copy.setPos(self.posStart, self.posEnd)
		return copy

	def __repr__(self):
		return f"<build-in function {self.name}>"

	#########################################

	def executePrint(self, executeContext):
		print(str(executeContext.symbolTable.get('value')))
		return RTResult().success(Number.null)
	executePrint.argNames = ['value']

	def executePrintRet(self, executeContext):
		return RTResult().success(String(str(executeContext.symbolTable.get('value'))))
	executePrintRet.argNames = ['value']

	def executeInput(self, executeContext):
		text = input()
		return RTResult().success(String(text))
	executeInput.argNames = []

	def executeInputInt(self, executeContext):
		while True:
			text = input()
			try:
				number = int(text)
				break
			except ValueError:
				print(f"'{text}' must be an integer. Try again!")
		return RTResult().success(Number(number))
	executeInputInt.argNames = []

	def executeClear(self, executeClear):
		os.system('cls' if os.name == 'nt' else 'cls')
		return RTResult().success(Number.null)
	executeClear.argNames = []

	def executeIsNumber(self, executeContext):
		isNumber = isinstance(executeContext.symbolTable.get("value"), Number)
		return RTResult().success(Number.true if isNumber else Number.false)
	executeIsNumber.argNames = ["value"]

	def executeIsString(self, executeContext):
		isNumber = isinstance(executeContext.symbolTable.get('value'), String)
		return RTResult().success(Number.true if isNumber else Number.false)
	executeIsString.argNames = ['value']

	def executeIsList(self, executeContext):
		isNumber = isinstance(executeContext.symbolTable.get("value"), List)
		return RTResult().success(Number.true if isNumber else Number.false)
	executeIsList.argNames = ['value']

	def executeIsFunction(self, executeContext):
		isNumber = isinstance(executeContext.symbolTable.get("value"), BaseFunction)
		return RTResult().success(Number.true if isNumber else Number.false)
	executeIsFunction.argNames = ['value']

	def executeAppend(self, executeContext):
		list_ = executeContext.symbolTable.get("list")
		value = executeContext.symbolTable.get("value")

		if not isinstance(list_, List):
			return RTResult().failure(RTError(self.posStart, self.posEnd, "First argument must be a list", executeContext))

		list_.elements.append(value)
		return RTResult().success(Number.null)
	executeAppend.argNames = ["list", "value"]

	def executePop(self, executeContext):
		list_ = executeContext.symbolTable.get("list")
		index = executeContext.symbolTable.get("index")

		if not isinstance(list_, List):
			return RTResult().failure(RTError(self.posStart, self.posEnd, "First argument must be a list", executeContext))

		if not isinstance(index, Number):
			return RTResult().failure(RTError(self.posStart, self.posEnd, "Second argumnet must be a number", executeContext))

		try:
			element = list_.elements.pop(index.value)
		except:
			return RTResult().failure(RTError(self.posStart, self.posEnd, "Element at this index could not be removed from list because index is out of bounds", executeContext))

		return RTResult().success(element)
	executePop.argNames = ["list", "index"]

	def executeExtend(self, executeContext):
		listA = executeContext.symbolTable.get("listA")
		listB = executeContext.symbolTable.get("listB")

		if not isinstance(listA, List):
			return RTResult().failure(RTError(self.posStart, self.posEnd, "First argument must be a list", executeContext))

		if not isinstance(listB, List):
			return RTResult().failure(RTError(self.posStart, self.posEnd, "Second argument must be a list", executeContext))

		listA.elements.extends(listB.elements)
		return RTResult().success(Number.null)
	executeExtend.argNames = ["listA", "listB"]

BuiltInFunction.print = BuiltInFunction("print")
BuiltInFunction.printRet = BuiltInFunction("printRet")
BuiltInFunction.input = BuiltInFunction("input")
BuiltInFunction.inputInt = BuiltInFunction("inputInt")
BuiltInFunction.clear = BuiltInFunction("clear")
BuiltInFunction.isNumber = BuiltInFunction("isNumber")
BuiltInFunction.isString = BuiltInFunction("isString")
BuiltInFunction.isList = BuiltInFunction("isList")
BuiltInFunction.isFunction = BuiltInFunction("isFunction")
BuiltInFunction.append = BuiltInFunction("append")
BuiltInFunction.pop = BuiltInFunction("pop")
BuiltInFunction.extend = BuiltInFunction("extend")

###########
# CONTEXT #
###########

class Context:
	def __init__(self, displayName, parent = None, parentEntryPos = None):
		self.displayName = displayName
		self.parent = parent
		self.parentEntryPos = parentEntryPos
		self.symbolTable = None

################
# SYMBOL TABLE #
################
class SymbolTable:
	def __init__(self, parent = None):
		self.symbols = {}
		self.parent = parent

	def get(self, name):
		value = self.symbols.get(name, None)
		if value == None and self.parent:
			return self.parent.get(name)
		return value

	def set(self, name, value):
		self.symbols[name] = value

	def remove(self, name):
		del self.symbols[name]

###############
# INTERPRETER #
###############

class Interpreter:
	def visit(self, node, context):
		methodName = f'visit{type(node).__name__}'
		method = getattr(self, methodName, self.noVisitMethod)
		return method(node, context)

	def noVisitMethod(self, node, context):
		raise Exception(f'No visit{type(node).__name__} method defined')

	################ VISIT NODES ####################

	def visitNumberNode(self, node, context):
		return RTResult().success(Number(node.tok.value).setContext(context).setPos(node.posStart, node.posStart))

	def visitStringNode(self, node, context):
		return RTResult().success(String(node.tok.value).setContext(context).setPos(node.posStart, node.posStart))

	def visitListNode(self, node, context):
		res = RTResult()
		elements = []

		for elementNode in node.elementNodes:
			elements.append(res.register(self.visit(elementNode, context)))
			if res.shouldReturn():
				return res

		return res.success(List(elements).setContext(context).setPos(node.posStart, node.posEnd))

	def visitVarAccessNode(self, node, context):
		res = RTResult()
		varName = node.varNameTok.value
		value = context.symbolTable.get(varName)

		if not value:
			return res.failure(RTError(node.posStart, node.posEnd, f"'{varName}' is not defined", context))

		value = value.copy().setPos(node.posStart, node.posEnd).setContext(context)
		return res.success(value)

	def visitVarAssignNode(self, node, context):
		res = RTResult()
		varName = node.varNameTok.value
		value = res.register(self.visit(node.valueNode, context))
		if res.shouldReturn():
			return res
		
		context.symbolTable.set(varName, value)
		return res.success(value)

	def visitBinOpNode(self, node, context):
		res = RTResult()
		left = res.register(self.visit(node.leftNode, context))
		if res.shouldReturn():
			return res
		right = res.register(self.visit(node.rightNode, context))
		if res.shouldReturn():
			return res

		if node.opTok.type == TT_PLUS:
			result, error = left.addedTo(right)
		elif node.opTok.type == TT_MINUS:
			result, error = left.subtractedBy(right)
		elif node.opTok.type == TT_MULTIPLY:
			result, error = left.multipliedBy(right)
		elif node.opTok.type == TT_DIVIDE:
			result, error = left.dividedBy(right)
		elif node.opTok.type == TT_POWER:
			result, error = left.poweredBy(right)
		elif node.opTok.type == TT_EQUALEQUAL:
			result, error = left.getComparisonEqual(right)
		elif node.opTok.type == TT_NOTEQUAL:
			result, error = left.getComparisonNotEqual(right)
		elif node.opTok.type == TT_LESSTHAN:
			result, error = left.getComparisonLessThan(right)
		elif node.opTok.type == TT_GREATERTHAN:
			result, error = left.getComparisonGreaterThan(right)
		elif node.opTok.type == TT_LESSTHANEQUAL:
			result, error = left.getComparisonLessThanEqual(right)
		elif node.opTok.type == TT_GREATERTHANEQUAL:
			result, error = left.getComparisonGreaterThanEqual(right)
		elif node.opTok.matches(TT_KEYWORD, 'and'):
			result, error = left.andedBy(right)
		elif node.opTok.matches(TT_KEYWORD, 'or'):
			result, error = left.oredBy(right)

		if error:
			return res.failure(error)
		else:
		   return res.success(result.setPos(node.posStart, node.posEnd))

	def visitUnaryOpNode(self, node, context):
		res = RTResult()
		number = res.register(self.visit(node.node, context))
		if res.shouldReturn():
			return res
		
		error = None

		if node.opTok.type == TT_MINUS:
			number, error = number.multipliedBy(Number(-1))
		elif node.opTok.matches(TT_KEYWORD, 'not'):
			number, error = number.notted()

		if error:
			return res.failure(error)
		else:
			return res.success(number.setPos(node.posStart, node.posEnd))

	def visitIfNode(self, node, context):
		res = RTResult()

		for condition, expr, shouldReturnNull in node.cases:
			conditionValue = res.register(self.visit(condition, context))
			if res.shouldReturn():
				return res

			if conditionValue.isTrue():
				exprValue = res.register(self.visit(expr, context))
				if res.shouldReturn():
					return res
				return res.success(Number.null if shouldReturnNull else exprValue)

		if node.elseCase:
			expr, shouldReturnNull = node.elseCase
			exprValue = res.register(self.visit(expr, context))
			if res.shouldReturn():
				return res
			return res.success(Number.null if shouldReturnNull else exprValue)

		return res.success(Number.null)

	def visitForNode(self, node, context):
		res = RTResult()
		elements = []

		startValue = res.register(self.visit(node.startValueNode, context))
		if res.shouldReturn():
			return res

		endValue = res.register(self.visit(node.endValueNode, context))
		if res.shouldReturn():
			return res

		if node.stepValueNode:
			stepValue = res.register(self.visit(node.stepValueNode, context))
			if res.shouldReturn():
				return res
		else:
			stepValue = Number(1)

		i = startValue.value
		
		if stepValue.value >= 0:
			condition = lambda: i < endValue.value
		else:
			condition = lambda: i > endValue.value

		while condition():
			context.symbolTable.set(node.varNameTok.value, Number(i))
			i += stepValue.value

			value = res.register(self.visit(node.bodyNode, context))
			if res.shouldReturn() and res.loopShouldContinue == False and res.loopShouldBreak == False:
				return res

			if res.loopShouldContinue:
				continue

			if res.loopShouldBreak:
				break

			elements.append(value)

		return res.success(Number.null if node.shouldReturnNull else List(elements).setContext(context).setPos(node.posStart, node.posEnd))

	def visitWhileNode(self, node, context):
		res = RTResult()
		elements = []

		while True:
			condition = res.register(self.visit(node.conditionNode, context))
			if res.shouldReturn():
				return res
			
			if not condition.isTrue():
				break

			value = res.register(self.visit(node.bodyNode, context))
			if res.shouldReturn() and res.loopShouldContinue == False and res.loopShouldBreak == False:
				return res

			if res.loopShouldContinue:
				continue

			if res.loopShouldBreak:
				break
			
			elements.append(value)

		return res.success(Number.null if node.shouldReturnNull else List(elements).setContext(context).setPos(node.posStart, node.posEnd))

	def visitFuncDefNode(self, node, context):
		res = RTResult()

		funcName = node.varNameTok.value if node.varNameTok else None
		bodyNode = node.bodyNode
		argNames = [argName.value for argName in node.argNameToks]
		funcValue = Function(funcName, bodyNode, argNames, node.shouldAutoReturn).setContext(context).setPos(node.posStart, node.posEnd)

		if node.varNameTok:
			context.symbolTable.set(funcName, funcValue)

		return res.success(funcValue)

	def visitCallNode(self, node, context):
		res = RTResult()
		args = []

		valueToCall = res.register(self.visit(node.nodeToCall, context))
		if res.shouldReturn():
			return res
		valueToCall = valueToCall.copy().setPos(node.posStart, node.posEnd)

		for argNode in node.argNodes:
			args.append(res.register(self.visit(argNode, context)))
			if res.shouldReturn():
				return res

		returnValue = res.register(valueToCall.execute(args))
		if res.shouldReturn():
			return res
		returnValue = returnValue.copy().setPos(node.posStart, node.posEnd).setContext(context)
		return res.success(returnValue)

	def visitReturnNode(self, node, context):
		res = RTResult()

		if node.nodeToReturn:
			value = res.register(self.visit(node.nodeToReturn, context))
			if res.shouldReturn():
				return res
		else:
			value = Number.null

		return res.successReturn(value)

	def visitContinueNode(self, node, context):
		return RTResult().successContinue()

	def visitBreakNode(self, node, context):
		return RTResult().successBreak()

#######
# RUN #
#######

globalSymbolTable = SymbolTable()
globalSymbolTable.set("null", Number.null)
globalSymbolTable.set("false", Number.false)
globalSymbolTable.set("true", Number.true)
globalSymbolTable.set("mathPI", Number.mathPI)
globalSymbolTable.set("print", BuiltInFunction.print)
globalSymbolTable.set("printRet", BuiltInFunction.printRet)
globalSymbolTable.set("input", BuiltInFunction.input)
globalSymbolTable.set("inputInt", BuiltInFunction.inputInt)
globalSymbolTable.set("clear", BuiltInFunction.clear)
globalSymbolTable.set("cls", BuiltInFunction.clear)
globalSymbolTable.set("isNum", BuiltInFunction.isNumber)
globalSymbolTable.set("isStr", BuiltInFunction.isString)
globalSymbolTable.set("isList", BuiltInFunction.isList)
globalSymbolTable.set("isFunc", BuiltInFunction.isFunction)
globalSymbolTable.set("append", BuiltInFunction.append)
globalSymbolTable.set("pop", BuiltInFunction.pop)
globalSymbolTable.set("extend", BuiltInFunction.extend)


def run(fName, text):
	lexer = Lexer(fName, text)
	tokens, error = lexer.makeTokens()
	if error:
		return None, error

	parser = Parser(tokens)
	ast = parser.parse()
	if ast.error:
		return None, ast.error

	interpreter = Interpreter()
	context = Context('<program>')
	context.symbolTable = globalSymbolTable
	result = interpreter.visit(ast.node, context)

	return result.value, result.error