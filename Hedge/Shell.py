import Hedge

while True:
	text = input('Hedge > ')
	if text.strip() == "":
		continue
	result, error = Hedge.run('<stdin>', text)

	if (error):
		print(error.asString())
	elif result:
		if len(result.elements) == 1:
			print(repr(result.elements[0]))
		else:
			print(repr(result))