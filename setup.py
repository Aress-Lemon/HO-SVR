from setuptools import setup, find_packages

setup(
    name="IOT",
    version="1.0",
    author="Aress",
    author_email="20228132027@m.scnu.edu.cn",
    description="工业互联网期末源码（HO-SVR）",

    # 项目主页（目前没有）
    # url="http://github.com/",

    # 程序分类信息
    classifiers=[
            "Programming Language :: Python :: 3.12",  # 编译版本
            "License :: OSI Approved :: MIT License", # 许可证信息
            "Operating System :: OS Independent", # 支持所有操作系统
        ],

    # 你要安装的包，通过 setuptools.find_packages 找到当前目录下有哪些包
    packages=find_packages()
)