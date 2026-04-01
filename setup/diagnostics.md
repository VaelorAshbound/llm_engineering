# Gathering Essential Diagnostic information

## Please run this next cell to gather some important data

Please run the next cell; it should take a minute or so to run (mostly the network test).
Then email me the output of the last cell to ed@edwarddonner.com.  
Alternatively: this will create a file called report.txt - just attach the file to your email.


```python
# Run my diagnostics report to collect key information for debugging
# Please email me the results. Either copy & paste the output, or attach the file report.txt

!pip install -q requests speedtest-cli psutil setuptools
from diagnostics import Diagnostics
Diagnostics().run()
```
