# 使用Docker：

### Container：
###### 容器，镜像运行后的实例状态，隔绝了本地环境和开发环境。
* 访问Docker内部网页需要在运行容器时通过:`-p 8080:8080` 添加端口映射。
### Image：
###### 镜像，运行用的包文件，如Ubuntu。
* <code>docker search ubuntu:14.04</code> 查找线上镜像，如ubuntu，14.04版本。
* <code>docker pull ubuntu:14.04</code> 下载镜像。
* <code>docker images</code> ，或者切换到界面工具中my images 查看本地镜像，可看到镜像名称、id等信息。
* <code>docker exec [container] /bin/bash</code> ，或者通过my images点击start和exec 运行镜像。之后若是系统类镜像，将进入到系统终端，先进行开发运行环境的配置，
* <code>docker ps 查看容器（提交时需要容器名称，在此命令下复制）
* <code>docker commit [OPTIONS] CONTAINER [REPOSITORY[:TAG]]</code> 提交对镜像的更改，用于配置自己的个性化镜像（注意镜像名称不支持大写字母）。如 <code>docker commit -t devcontainerforlw imageforfrontbylw:lastest</code> 。
* <code>docker export > develop.tar</code> 当前Container导出Image。执行完后，在你的个人目录下（Mac 上是 /Users/你的用户名）可以找到`develop.tar`文件。
### Volume：
###### 数据卷，即可指定读取存储外部数据的路径，如项目所在文件夹。
* 启动Image时可以通过 `-v 本地目录:容器目录`映射本地路径到容器，一个容器可以挂载多个Volume，Volume中的数据独立于Image。

### 配置一个自己的镜像：
1. 下载并安装Docker，顺带安装kitematic界面工具方便操作，配置加速器；
2. 注册一个Docker Hub账号并登陆，下载并运行镜像，如ubuntu系统；
3. 通过命令行做常规的系统初始化工作（换源、安装常用工具）
4. 通过命令行安装开发环境，如前端npm=》nodejs=》webpakc=》vue-cli等（开发工具仍安装在本地环境中），之后记得提交镜像方可保存。
5. 导出镜像 docker export

### 使用自己的镜像流程:
1. 准备好Docker安装包和镜像；
2. 安装Docker、打开Kitematic，注册账号并登陆；
3. 打开Docker命令行：在 Kitematic 中点击左下角“DOCKER CLI”；
4. 导入镜像：输入命令docker import，从文件夹中直接把 ubuntu 文件拖拽到命令行中（注意文件路径中不能有中文，如果有，先把文件移动到另一个纯英文路径的文件夹中）；
5. 挂载本地文件夹和映射访问地址并运行镜像：输入命令docker images，复制出镜像的 IMAGE ID（如54184b993d46）。输入命令,docker run -t  -d -p 8080:8080/tcp  --name dev -v /local/path/to/workplace:/container/path/to/workplace IMAGEID （IMAGEID为上一步复制的）

#### 参数介绍：
* `--name` 自定义容器名称
* `-d` 后台运行
* `-p` 本地端口:容器端口 本地端口映射到容器端口
* `-v` 本地文件夹:容器内部路径 挂载本地文件夹到容器内部路径

记录配置命令和映射等内容方便日后重做和查看：使用dockerfile
打开终端，进入我们指定的文件目录，新建Dockerfile文件，把做过的指令写下来前面加个RUN
（apt-get install后面需要多加-y，可在之后需要交互的地方默认Yes，否则之后的建立镜像时，linux apt-get会因为得不到用户响应而自动退出。
之后即可以通过运行dockerfile文件直接建立镜像

### 总结：
docker运行镜像生成container，在其中安装开发环境并保存，本地装开发工具和放置项目，通过挂载项目到images中运行，通过映射端口在浏览器中访问container中运行的项目。