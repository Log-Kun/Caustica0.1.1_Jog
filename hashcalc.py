import hashlib

def dist_name(url: str) -> str:
    digest = hashlib.md5(url.encode('utf-8')).digest()
    n = int.from_bytes(digest, 'big')
    if n == 0:
        return '0'
    chars = '0123456789abcdefghijklmnopqrstuvwxyz'
    out = ''
    while n:
        n, r = divmod(n, 36)
        out = chars[r] + out
    return out

old = 'file:///D:/Desktop/gradle-9.7.1-bin.zip'
new = 'https://services.gradle.org/distributions/gradle-9.7.1-bin.zip'
print('old ->', dist_name(old))
print('new ->', dist_name(new))