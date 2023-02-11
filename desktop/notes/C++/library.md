@[TOC](C++基础-如何引入第三方静态库、动态库或自定义库)

# 静态库(.lib)

无论库是否有开源，最终能通过生成或下载拿到静态库文件.lib和头文件.h即可使用静态库引入方式。步骤如下：

## 已编译好的库

使用glfw作为示例[官方下载地址](https://www.glfw.org)：

1. 将下载好的静态库文件glfw.lib和glfw.h按照想要的路径放置到解决方案里，例如【解决方案】\Dependencies\glfw\文件夹里；
2. 在VS开发工具里打开应用程序项目属性（右键打开），找到C++》常规》附加包含目录，添加头文件所在基础路径`$(SolutionDir)\Dependencies\glfw\;`，注意不要删除原本附加包含目录里已有的路径，只使用分号隔开，除非你确定那些路径已经不需要；
3. 接着上一步，重新在属性里找链接器》常规》附加库目录，添加静态库文件所在基础路径`$(SolutionDir)\Dependencies\glfw\;`。之后找到链接器》输入》附加依赖项，添加静态库剩余路径`glfw3.lib`，基础路径和剩余路径合起来才是静态库的完整路径`$(SolutionDir)\Dependencies\glfw\glfw3.lib`；
4. 在需要使用该库的项目文件代码里引入头文件，路径根据第2步基础路径所决定，如`#include <glfw3.h>`指向的是`$(SolutionDir)\Dependencies\glfw\glfw3.h`；
5. 现在可以开始调用静态库里的方法了，代码示例：
```bash
#include <iostream>
#include <glfw3.h>
int main() {
  int a = glfwInit();
  std::cout << a << std::endl;
}
```

## 引用依赖库
适用于有源代码的库，例如第三方开源库，或者自己的库。
可以不放到当前解决方案里，直接作为单独项目生成静态库文件和头文件，然后按照上述方式引入静态库，也可以按照下面的方法，将源代码放到解决方案里作为单独一个库项目，让应用程序项目引入，这种方式的好处是根据应用程序需要修改库代码。
还是使用上一步有开源的glfw作为示例：
1. 将下载好的库源代码放置到解决方案里，如放到【解决方案】\glfw\文件夹里；
2. 在VS开发工具里打开该项目属性，找到常规》配置类型，将其选择为静态库(.lib)，完成后按顺序点击下方应用Apply、确定OK按钮；
3. 在VS开发工具里打开应用程序项目属性，找到C++》常规》附加包含目录，添加头文件所在基础路径`$(SolutionDir)\glfw\src\;`；
4. 接着在VS开发工具应用程序项目打开引用栏（项目右键选择添加》引用），里面可以看到当前解决方案下所有库项目，包括我们第2步配置glfw库项目的名称和路径，在想要依赖的库项目前面打钩，并点击下方确定OK；
5. 在需要使用该库的项目文件代码里引入头文件glfw3.h，路径根据第2步基础路径所决定，如`#include <glfw3.h>`指向的是`$(SolutionDir)\glfw\src\glfw3.h`；
6. 现在可以开始调用引用依赖库的方法了，代码示例：
```bash
#include <iostream>
#include <glfw3.h>
int main() {
  int a = glfwInit();
  std::cout << a << std::endl;
}
```

# 动态库(.dll)
适用于第三方库为主，虽然也可以用于自己的库项目，但考虑到静态库比动态库总体积更小更高效率来说，自己的库项目还是直接放到解决方案里添加引用依赖即可。
依旧使用glfw动态库作为示例：
1. 将下载好的动态库glfw.dll、glfw3dll.lib（动态库专用的链接文件）和glfw.h放到依赖下`$(SolutionDir)\Dependencies\glfw\`；
2. 在VS开发工具里打开应用程序项目属性，找到C++》常规》附加包含目录，添加头文件所在基础路径`$(SolutionDir)\Dependencies\glfw\;`；
3. 接着上一步，重新在属性里找链接器》常规》附加库目录，添加动态库文件所在基础路径`$(SolutionDir)\Dependencies\glfw\;`。之后找到链接器》输入》附加依赖项，添加静态库剩余路径`glfw3dll.lib`，基础路径和剩余路径合起来才是静态库的完整路径`$(SolutionDir)\Dependencies\glfw\glfw3dll.lib`；
4. 将动态库glfw.dll放到应用程序所生成的exe文件的旁边（否则点击exe直接运行时会报错找不到glfw.dll库；
5. 在需要使用该库的项目文件代码里引入头文件，路径根据第2步基础路径所决定，如`#include <glfw3.h>`指向的是`$(SolutionDir)\Dependencies\glfw\glfw3.h`；
6. 现在可以开始调用静态库里的方法了，代码示例：
```bash
#include <iostream>
#include <glfw3.h>
int main() {
  int a = glfwInit();
  std::cout << a << std::endl;
}
```
