import inspect


def format_function(func):
    """Formats a function in a descriptive way, especially for lambda functions.
    
    Args:
        func: A callable function
        
    Returns:
        str: A descriptive string representation of the function
    """
    # If it's a lambda function, try to get its source code
    if func.__name__ == '<lambda>':
        try:
            source = inspect.getsource(func)
            # Clean up the source: remove leading/trailing whitespace and newlines
            source = source.strip()
            # Try to extract just the lambda expression part
            # If it's on multiple lines, take the first meaningful line
            lines = source.split('\n')
            # Find the line containing 'lambda'
            for line in lines:
                if 'lambda' in line:
                    # Clean up the line
                    cleaned = line.strip()
                    # Remove any leading assignment or context
                    if '=' in cleaned and 'lambda' in cleaned:
                        # Extract the part after the = sign
                        parts = cleaned.split('=', 1)
                        if len(parts) > 1:
                            cleaned = parts[1].strip()
                    # Limit length for readability
                    if len(cleaned) > 80:
                        cleaned = cleaned[:77] + '...'
                    return cleaned
            # Fallback: return first line if no lambda found
            return lines[0].strip() if lines else str(func)
        except (OSError, TypeError):
            # If we can't get source, fall back to a descriptive name
            # Try to get the qualname which might have more context
            qualname = getattr(func, '__qualname__', None)
            if qualname and qualname != '<lambda>':
                return f"<lambda in {qualname}>"
            return "<lambda function>"
    else:
        # For named functions, use their name
        qualname = getattr(func, '__qualname__', func.__name__)
        module = getattr(func, '__module__', None)
        if module and module != '__main__':
            return f"{module}.{qualname}"
        return qualname


def const(x):
    return lambda _: x


def fst(x):
    return x[0]


head = fst


def tail(x):
    return x[1:]


def snd(x):
    return x[1]


def validcheck(f=None):
    if f:
        return lambda x: f(x) is not None
    else:
        return lambda x: x is not None
