import re

# Pseudocode definitions
pseudocode = """
// Pseudoscalars
IE1 = { e1^e2^e3 }
IE2 = { e6^e7^e8 }
IC1 = { e1^e2^e3^e4^e5 }
IC2 = { e6^e7^e8^e9^e10 }
ID =  { e1^e2^e3^e4^e5^e6^e7^e8^e9^e10 }

// DCGA dualization macro
DD =  { -_P(1).ID() }
// CGA1 and CGA2 dualizations
C1D = { -_P(1).IC1() }
C2D = { -_P(1).IC2() }
// Euclidean 1 and Euclidean 2 dualizations
E1D = { -_P(1).IE1() }
E2D = { -_P(1).IE2() }
// special Dual macro for DCGA dualization operator *
Dual = { DD(_P(1)) }
"""

# Regular expressions to parse pseudocode
pattern_scalar = r'^([A-Z0-9]+)\s*=\s*{([^}]*)}'
pattern_dual = r'^([A-Z0-9]+)\s*=\s*{([^}]*)\(_P\(1\)\)}'

# Dictionary to store mappings
functions = {}

# Function to create the multivector components
def multivector(*components):
    return ' ^ '.join(components)

# Function to create Python function definitions
def create_function(name, expression):
    return f"def {name}():\n    return {expression}\n"

# Parse each line of pseudocode
lines = pseudocode.strip().split('\n')
for line in lines:
    line = line.strip()
    if line.startswith('//') or not line:
        continue
    
    match_scalar = re.match(pattern_scalar, line)
    match_dual = re.match(pattern_dual, line)
    
    if match_scalar:
        name = match_scalar.group(1)
        expression = match_scalar.group(2)
        functions[name] = create_function(name, f"multivector({expression})")
    elif match_dual:
        name = match_dual.group(1)
        operation = match_dual.group(2)
        functions[name] = create_function(name, f"-_P(1).{operation}()")
    else:
        raise ValueError(f"Unable to parse line: {line}")

# Generate output
for name, func_definition in functions.items():
    print(func_definition)
