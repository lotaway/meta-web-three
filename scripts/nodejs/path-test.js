/**
 * Created by lw on 2016/6/1.
 */
var path = require('path');
//path的模块，可以帮你标准化，连接，解析路径，从绝对路径转换到相对路径，从路径中提取各部分信息，检测文件是否存在。总的来说，path模块其实只是些字符串处理，而且也不会到文件系统去做验证（path.exists函数例外）。

//  路径的标准化
//  在存储或使用路径之前将它们标准化通常是个好主意。比如，由用户输入或者配置文件获得的文件路径，或者由两个或多个路径连接起来的路径，一般都应该被标准化。可以用path模块的normalize函数来标准化一个路径，而且它还能处理“..”，“.”“//”。比如：
path.normalize('/foo/bar//baz/asdf/quux/..');   // '/foo/bar/baz/asdf'

//  连接路径
//  使用path.join()函数，可以连接任意多个路径字符串，只用把所有路径字符串依次传递给join()函数就可以：
path.join('/foo', 'bar', 'baz/asdf', 'quux', '..'); // '/foo/bar/baz/asdf'

//  解析路径
//  用path.resolve()可以把多个路径解析为一个绝对路径。它的功能就像对这些路径挨个不断进行“cd”操作，和cd命令的参数不同，这些路径可以是文件，并且它们不必真实存在——path.resolve()方法不会去访问底层文件系统来确定路径是否存在，它只是一些字符串操作。比如：
path.resolve('/foo/bar', './baz');  // /foo/bar/baz
path.resolve('/foo/bar', '/tmp/file/'); // /tmp/file

