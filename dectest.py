from decorator import decorator


def test(pr="HOAL"):
    x = [None]
    @decorator
    def _test(f, *args, **kwargs):
        if x[0] is None:
            x[0] = "hola"
            print(pr)

        return f(*args, **kwargs)

    return _test

@test(pr="SIII!!!")
def fun(x, y):
    print "chao"

fun(1, 2)
fun(1, 2)
fun(1, 2)
