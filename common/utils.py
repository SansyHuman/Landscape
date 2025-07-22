import math

def prime_numbers(n: int) -> list[int]:
    if n <= 0:
        return []
    primes = [2]
    for i in range(1, n):
        prime = primes[-1] + 1
        while True:
            is_prime = True
            for p in primes:
                if p > math.sqrt(prime):
                    break
                if prime % p == 0:
                    is_prime = False
                    break

            if is_prime:
                primes.append(prime)
                break

            prime += 1

    return primes