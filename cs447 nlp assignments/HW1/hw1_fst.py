from fst import *

# here are some predefined character sets that might come in handy.
# you can define your own
AZ = set("abcdefghijklmnopqrstuvwxyz")
VOWS = set("aeiou")
CONS = set("bcdfghjklmnprstvwxyz")
FINAL_CONS_DOUBLE = set("nptr")
E = set("e")
I = set("i")
U = set("u")
CONS_U = CONS.union(U)

# Implement your solution here
def buildFST():
    #
    # The states (you need to add more)
    # ---------------------------------------
    # 
    f = FST("q0") # q0 is the initial (non-accepting) state

    f.addState("i")
    f.addState("iPrime")
    f.addState("ie")
    f.addState("iePrime")

    f.addState("u")
    f.addState("ue")
    f.addState("uePrime")

    f.addState("cons")
    f.addState("consPrime")
    f.addState("consao")
    f.addState("conse")
    f.addState("consi")
    f.addState("consie")
    f.addState("consu")
    f.addState("consue")

    f.addState("qing")
    f.addState("q_EOW", True) # an accepting state (you shouldn't need any additional accepting states)

    #
    # The transitions:
    # ---------------------------------------
    f.addSetTransition("q0",AZ-I-CONS_U,"q0")
    f.addTransition("q0","", "ing","q_EOW")
    f.addTransition("qing","", "ing","q_EOW")

    # ie => ying
    f.addTransition("q0","i","y","i")
    f.addTransition("q0","i","i","iPrime")
    f.addSetTransition("iPrime",AZ-CONS_U-E-I,"q0")
    f.addTransition("iPrime","e","e","iPrime")
    f.addTransition("iPrime","i","y","i")
    f.addTransition("iPrime","i","i","iPrime")
    f.addTransition("iPrime","e","e","iePrime")
    f.addTransition("iePrime","i","i","iPrime")
    f.addTransition("iePrime","i","y","i")
    f.addSetTransition("iePrime",AZ-CONS_U-I,"q0")
    f.addTransition("i","e","","ie")
    f.addTransition("ie","","ing","q_EOW")

    # [u]e => [u]ing
    f.addSetTransition("q0", U, "u")
    f.addSetTransition("iPrime", U, "u")
    f.addSetTransition("iePrime", U, "u")
    f.addSetTransition("u", U, "u")
    f.addSetTransition("u",AZ-I-CONS_U-E,"q0")
    f.addTransition("u", "e", "e", "uePrime")
    f.addTransition("u", "e", "", "ue")
    f.addTransition("u", "i", "i", "iPrime")
    f.addTransition("u", "i", "y", "i")
    f.addTransition("u", "", "ing", "q_EOW")
    f.addTransition("ue", "", "ing", "q_EOW")
    f.addSetTransition("uePrime", U, "u")
    f.addSetTransition("uePrime", AZ-I-CONS_U, "q0")
    f.addTransition("uePrime", "i", "i", "iPrime")
    f.addTransition("uePrime", "i", "y", "i")

    # cons vowel nptr, [cons]e => [cons]ing
    f.addSetTransition("q0",CONS,"cons")
    f.addSetTransition("iPrime", CONS, "cons")
    f.addSetTransition("iePrime", CONS, "cons")
    f.addSetTransition("u", CONS, "cons")
    f.addSetTransition("uePrime", CONS, "cons")
    f.addSetTransition("cons", CONS, "cons")
    f.addSetTransition("cons", VOWS-E-I-U, "consao")
    f.addTransition("cons", "e", "", "qing")
    f.addTransition("cons", "e", "e", "conse")
    f.addTransition("cons", "i", "i", "consi")
    f.addTransition("cons", "i", "y", "i")
    f.addTransition("cons", "u", "u", "consu")
    f.addTransition("cons", "", "ing", "q_EOW")

    f.addTransition("consao", "i", "i", "iPrime")
    f.addTransition("consao", "i", "y", "i")
    f.addSetTransition("consao", CONS-FINAL_CONS_DOUBLE, "cons")
    f.addSetTransition("consao", FINAL_CONS_DOUBLE, "consPrime")
    f.addTransition("consao", "n", "nn", "qing")
    f.addTransition("consao", "p", "pp", "qing")
    f.addTransition("consao", "t", "tt", "qing")
    f.addTransition("consao", "r", "rr", "qing")
    f.addSetTransition("consao", VOWS-I-U, "q0")
    f.addTransition("consao", "u", "u", "u")

    f.addTransition("conse", "i", "i", "iPrime")
    f.addTransition("conse", "i", "y", "i")
    f.addSetTransition("conse", CONS-FINAL_CONS_DOUBLE, "cons")
    f.addSetTransition("conse", FINAL_CONS_DOUBLE, "consPrime")
    f.addTransition("conse", "n", "n", "qing")
    f.addTransition("conse", "p", "pp", "qing")
    f.addTransition("conse", "t", "tt", "qing")
    f.addTransition("conse", "r", "r", "qing")
    f.addSetTransition("conse", VOWS-I-U, "q0")
    f.addTransition("conse", "u", "u", "u")

    f.addTransition("consi", "i", "i", "iPrime")
    f.addTransition("consi", "i", "y", "i")
    f.addTransition("consi", "e", "e", "consPrime")
    f.addSetTransition("consi", CONS-FINAL_CONS_DOUBLE, "cons")
    f.addSetTransition("consi", FINAL_CONS_DOUBLE, "consPrime")
    f.addTransition("consi", "n", "nn", "qing")
    f.addTransition("consi", "p", "pp", "qing")
    f.addTransition("consi", "t", "tt", "qing")
    f.addTransition("consi", "r", "rr", "qing")
    f.addSetTransition("consi", VOWS-E-I-U, "q0")
    f.addTransition("consi", "u", "u", "u")

    f.addTransition("consu", "i", "i", "iPrime")
    f.addTransition("consu", "i", "y", "i")
    f.addTransition("consu", "e", "", "ue")
    f.addTransition("consu", "e", "e", "consPrime")
    f.addSetTransition("consu", CONS-FINAL_CONS_DOUBLE, "cons")
    f.addSetTransition("consu", FINAL_CONS_DOUBLE, "consPrime")
    f.addTransition("consu", "n", "nn", "qing")
    f.addTransition("consu", "p", "pp", "qing")
    f.addTransition("consu", "t", "tt", "qing")
    f.addTransition("consu", "r", "rr", "qing")
    f.addSetTransition("consu", VOWS-E-I-U, "q0")
    f.addTransition("consu", "u", "u", "u")

    f.addSetTransition("consPrime", VOWS-E-I-U, "q0")
    f.addSetTransition("consPrime", CONS, "cons")
    f.addTransition("consPrime", "u", "u", "u")
    f.addTransition("consPrime", "e", "", "qing")
    f.addTransition("consPrime", "e", "e", "consPrime")
    f.addTransition("consPrime", "i", "i", "iPrime")
    f.addTransition("consPrime", "i", "y", "i")

    # Return your completed FST
    return f
    

if __name__ == "__main__":
    # Pass in the input file as an argument
    if len(sys.argv) < 2:
        print("This script must be given the name of a file containing verbs as an argument")
        quit()
    else:
        file = sys.argv[1]
    #endif

    # Construct an FST for translating verb forms 
    # (Currently constructs a rudimentary, buggy FST; your task is to implement a better one.
    f = buildFST()
    # Print out the FST translations of the input file
    f.parseInputFile(file)
