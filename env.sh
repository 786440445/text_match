pip3 config set global.index-url http://pypi.douban.com/simple/ 
pip3 config set install.trusted-host pypi.douban.com
 
pip3 install --upgrade pip

pip3 install tensorflow-gpu==1.15

pip3 install -r requirements.txt

# # http://pypi.douban.com/simple/
# # https://pypi.tuna.tsinghua.edu.cn/simple