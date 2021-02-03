doc = "For more details please mail us at: xyz@abc.com,pqr@mno.com"

import re

addresses = re.findall(r'[\w\.-]+@[\w\.-]+', doc)
for address in addresses:
    print(address)
