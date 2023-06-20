@[TOC](php学习笔记-安装踩坑之路)
曾经，我以为php这么简单易用的开发语言，配合Linux这么统一易用的包安装方式，肯定比npm安装第三方包还简单吧，然后就发现，我错了。

# 介绍

PHP是世界上最好的语言，Linux是世界上最好的系统。
以上都是我骗你的。
物理学不存在了。
反正简单地讲就是，我想着在自己电脑里运行Wordpress，那Wordpress需要Php环境才能用啊，于是我就想着在Window的子系统WLS里安装Php，想着作为Linux的WLS安装Php已经是5分钟能搞定的事情吧。
然后我就搞了50个小时还没解决。
接下来如果不怕被折磨，就跟随我的脚步一起进入这个折磨之旅。

# 下载
[Wordpress]()
[Php下载](https://www.php.net/downloads.php)

不知为何通过这种方式下载安装后并无法正常使用，虽然在命令行输入`php -version`能检测到php版本，但是缺少了php-rpm等很多东西，Nginx里配置的fastcgi没有可配置的服务路径。
之后直接通过Linux自身的包管理下载：
```bash
apt intall php php-rpm
```
这种方式成功安装了php8.1，虽然与官网直接下载的php8.2.4版本不一致，但奇怪的是两者竟然能共存，只不过之后会有什么问题我也不清楚。
目前的情况就是fastcgi配置的是php8.1里的rpm.sock，而命令行里查到的php是8.2.4，使用`echo $PATH`也没看到php配置的路径。