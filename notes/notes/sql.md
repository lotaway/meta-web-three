@(TOC)[SQL学习-基础课程]

# 概念

SQL是Structured Query Language 的缩写,意为“结构化查询语言”。一般代指现在计算机中的各种SQL数据库，如MySQL、MicrosoftServerSQL。

* 以下语句以MySQL为主，其他数据库在字段类型、语法上会有轻微区别，但大同小异。

# 创建

通过`create table [tableName] (Array<field>)`创建一个用户表：

```shell
--  用户表
create table `User` (
  id integer primary key unique auto_increment comment '用户标识',
  username varchar(20) not null unique comment '用户名',
  nickname varchar(20) default '' comment '昵称',
  sex char(2) not null default 0 comment '性别',
  age char(2) not null default 0 comment '年龄',
  authorId integer comment '作者标识',
  createTime Datetime(3) default Datetime(3),
  updateTime Datetime(3) default Datetime(3)
);
-- 作者表
create table `Author` (
  id integer primary key unique auto_increment comment '作者标识',
  nickname varchar(20) unique comment '昵称',
  userId integer not null,
  constraint `Author_userId_fkey` foreign key (userId) references `User`(`id`),
  state char(1) default 0 comment '账号状态'
);
-- 书籍表
create table `Book` (
  id integer primary key unique auto_increment comment '书标识',
  authorId integer not null,
  constraint `Book_authorId_fkey` foreign key (authorId) references `Author`(`id`),
  bookName varchar(30) not null unique comment '书名',
  state char(1) default 0 comment '书籍状态',
  readCount integer default 0 '阅读量'
)
-- 书籍章节内容
create table `BookChapter` (
  id varchar(50) unique default uuid() comment '书籍章节标识',
  bookId integer not null,
  constraint `BookChapter_bookId_fkey` foreign key references `Book`(`id`)
  chapterName varchar(50) not null unique comment '章节名称',
  state char(1) default 0 comment '章节状态',
  content text default '' comment '章节内容',
  readCount integer default 0 comment '阅读量'
)
-- 订阅书籍，通过state：【0：互不关注，1：用户1关注用户2，2：用户2关注用户1，3：用户1和2互相关注】
create table `AuthorSubscribe` (
  id integer unique default uuid() comment '订阅标识',
  fUserId integer not null,
  constraint `BookSubscribe_fUserId_fkey` foreign key (fUserId) references `Author`(`id`)
  sUserId integer not null,
  constraint `BookSubscribe_sUserId_fkey` foreign key (sUserId) references `Author`(`id`),
  state char(1) not null default 0
)
```

# 查询

通过语句`select * from [tableName]`查询

```shell
select * from User;
select * from Author;
select bookName,state from Book;
```

## 多表联查

通过在查询语句后面加入join，可细分为left join, right join, inner join, full join，默认为left join：

```shell
select User.*, Author.* from User join Author where User.authorId=Author.id
``` 

三表联查：

```shell
select User.*, Author.*, Book.* from (User join Author where User.authorId=Author.id) join Book where AUthor.id=Book.authorId
```

# 添加/插入数据

通过语句`insert into [tableName] values (Array<field>)`插入数据

# 修改/更新数据

通过语句`update [tableName] set [field]=[value] where [condition]`：

```shell
update `Author` set penname='笔下留情' where id=1;
```

# 删除数据

通过语句`delete from [tableName] where [condition]`：

```shell
delete from `BookChapter` where id=1;
```