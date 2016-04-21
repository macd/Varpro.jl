
FUZZ = 1.e-5

function isclose(x::Number, y::Number)
    if abs(x - y) < FUZZ
        return true
    else
        return false
    end
end


function isclose(x::Array, y::Array)
    if size(x) != size(y)
        return false
    end
    for i in length(x)
        if !isclose(x[i], y[i])
            return false
        end
    end
    return true
end

# OK, this is just so we get a consistent ordering on the complex
# numbers.  But we should think more carefully about just how robust
# this hack is.
import Base: isless
function isless(x::Complex, y::Complex)
    return abs(x) < abs(y)
end

