
doc = "For more details please mail us at: xyz@abc.com,pqr@mno.com"

import re

new_email_address = re.sub(r'([\w\.-]+)@([\w\.-]+)',r'dima@mno.com', doc)
print(new_email_address)

