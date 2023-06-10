# Linux启动过程

Linux按照以下步骤启动：
1. 主机加电硬件POST（Power-OnSelf-Test）自检，加载BIOS硬件信息；
2. 引导操作系统，即加载MBR（Master Boot Record）,引导GRUB（GRand Unified Bootloader）启动管理器；
3. 引导Linux内核，运行第一个进程init（进程号永远为1）；
4. 进入响应的运行级别；
5. 运行终端，启动登录管理器，等待输入用户名和密码后登录系统。

# Linux常用命令

`cd [路径/文件夹]`进入指定文件夹下，可包含路径，如/etc/nginx/sites-enabled，路径开头是斜杠/表示从根目录开始匹配，如果不带斜杆，而是直接文件夹或路径名称或点斜杠./开头则是相对于当前命令行所在的路径，若是../开头则是相当于上一级的路径。
`rm -rf [路径/文件夹]`表示删除指定文件夹及其内部所有文件
`vim [路径/文件]` 表示使用vim工具打开一个文件进行编辑，若该文件不存在则会打开一个新文件，保存后才会进行创建
`touch [路径/文件]` 表示以只读模式查看一个文件
`service [服务名] status/start/stop/restart`查看服务状态/开启/停止/重启服务
`systemctl status/restart/stop/enabled [服务名]`查看系统服务状态/重启/停止/允许开机启动
`find [文件名]`查找文件所在位置
`whereis [服务名]`查找服务所在位置
`cat [服务名]`查看服务所在路径