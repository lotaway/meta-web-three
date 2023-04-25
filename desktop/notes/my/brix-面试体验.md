@[TOC](背景介绍：Brix是一个国内管理x国外雇佣的远程办公平台，帮助入驻的工程师和有意愿的北美企业达成雇佣关系)

入驻Brix平台第一轮就是笔试，我算是发现了，反正万物总将回归算法，无论你之前做的是什么领域都好，也不管你打算做什么职位，最终的朝向就是数学、算法工程师。

# 笔试/算法题挑战

千篇一律的算法题挑战，共三道，时间一小时，难度可选简单/中等/困难，但我感觉没多大区别，直接上困难就行了。
个人感觉这个部分就是为了筛选掉完全不具备写代码能力的人，防止浪费人力面试。
由于我试过了中等和困难两个级别，因此总共有5道题可做，其中有一道重复了本来应该是2*3=6道。

## 二叉树检查（中等难度）

该题目要求对二叉搜索树进行检查，使得所有的左节点的值都小于它所有的父节点和祖父节点，所有右节点的值都大约它所有的父节点和祖父节点。
如：

```javascript
const root = {
    left: {
        left: {
            value: 1
        },
        value: 3,
        right: {
            value: 4
        }
    },
    value: 5,
    right: {
        value: 7,
        right: {
            value: 9
        }
    }
}
```

```javascript
function binarySearchTreeCheck(root) {
    function valid(parent, min, max) {
        if (min !== null && min >= parent.value) return false
        if (max !== null && parent.value <= max) return false
        return valid(parent.left, min, parent.value) && valid(parent.right, parent.value, max)
    }

    return valid(root, null, null)
}
```

# 矩阵转换（中等难度）

要求将
| 1 2 3 |
| 4 5 6 |
转换为
| 1 4 |
| 2 5 |
| 3 6 |
实际提供的是一个3列*2行二维数组，转换后即为2行*3列，题本身不难，难在做好效率拿满分，我只能出一个不是满分的答案：

```javascript
function matrixTranform(parens) {
    let array = [];
    parens.forEach((rows, rIndex) => {
        rows.forEach((num, nIndex) => {
            if (!array[nIndex]) array[nIndex] = [];
            array[nIndex][rIndex] = num;
        });
    });
    return array;
}
```

# 小括号检查（中等/困难）

要求检查所有左右括号是否正确匹配上，如"(abc)(as)"是正确的，而"()))"，")("是不正确的。
```javascript
function quoteCheck(parens) {
    //  先通过正则和字符串转数组的分隔符将不是括号的项和紧靠的括号对都排除掉
    let array = parens.replace(/[^\(\)|(\(\))+]+/g, "").split("()")
    let count = 0
    while (count !== array.length) {
        count = array.length
        array = array.join("").split("()")
    }
    array = array.join("").split("")
    //  接着是真正的匹配，通过记录左括号的已有数量来检查，如果在左括号数量不足时出现了右括号就代表这个字符串无法完全匹配所有括号
    let result = true
    let left = 0
    for (let i = 0, l = array.length; i < l; i++) {
        if (!result || left > l - i) {
            break;
        }
        switch (array[i]) {
            case ")":
                if (left === 0)
                    result = false
                else
                    --left
                break
            case "(":
                ++left
                break
        }
    }
    //  整个字符串都匹配完后，若左括号还有剩余也是错误的
    return result && left === 0
}
```

## 获取字符串中非重复字符长度（困难）

获取字符串里没有重复字符的子字符串长度，如abcacd中，abc是最长的无重复字符的子字符串，长度为3

```bash
int lengthOfLongestSubstring(std::string s) {
    size_t size = s.size();
    size_t longestCount = 0;
    int startIndex = 0;
    std::unordered_map<char, int> char2LastIndex;
    for (size_t i = startIndex; i < size; i++) {
        char c = s[i];
        auto charLastIndex = char2LastIndex.find(c);
        bool isLast = i == size - 1;
        bool hasRepeat = charLastIndex != char2LastIndex.end() && charLastIndex->second >= startIndex;
        //  if already have a repeat char inside the child string, just count get the longest count and reset the child start index
        if (isLast || hasRepeat) {
            longestCount = max(i - startIndex + (hasRepeat ? 0 : 1), longestCount);
            if (isLast)
                break;
            startIndex = charLastIndex->second + 1;
            if (size - startIndex <= longestCount)
                break;
        }
        char2LastIndex[c] = i;
    }
    return longestCount;
}
```

## 树结构转数组（困难）

这题和二叉树有点相似，都是对树结构的处理，和二叉树不同的是子节点可以有两个以上。
要求是按照层数排列，既顶层节点，所有2层节点，所有3层节点...

以下模拟树结构数据，要求转为[1, 2, 3, 4, 5, 6, 7, 8, 9]

```javascript
const mock = {
    value: 1, childNodes: [{
        value: 2, childNodes: [{
            value: 4, childNodes: [{
                value: 8, childNodes: null
            }, {
                value: 9, childNodes: null
            }]
        }, {
            value: 5, childNodes: null
        }, {
            value: 6, childNodes: null
        }]
    }, {
        value: 3, childNodes: [{
            value: 7, childNodes: null
        }]
    }]
}
```

既然是按照层数转换，就在递归时提供一个层数，每进入深一层递归就增加一层，完成一个按照层级的二维数组，之后再转换为一维数组，可能效率不是很好，但也是我目前想到的唯一解法。

```javascript
function tree2Array(root = mock) {
    const array = []
    const toArr = (target, level = 0) => {
        array[level] = array[level] || []
        array[level].push(target.value)
        if (target.childNodes === null) return false
        return target.childNodes.map(childNode => toArr(childNode, level + 1))
    }
    toArr(root)
    return array.reduce((prev, item) => {
        prev.push(...item)
        return prev
    }, [])
}
```

# 平台面试

之前遇到的面试官很喜欢问一些偏理论背诵的内容，例如React和Vue的区别，小程序和网页的区别，为什么用Redux、对Docker的了解之类。
这次遇到的不知道是做的工作比较偏还是其他原因，他直接拿了在线协作文档这种具体的项目与我展开讨论，询问一些技术实现的理论可能，例如怎么处理文档的频繁修改，多人协作查看和编辑同一篇好几M大小的文档应该怎么做，如何保证存储的顺序，微服务下库存系统和订单系统如何保证事务处理的原子性，Redis和Sql数据库如何保证数据一致性等。
感觉面试过程微妙的和谐，可能是做惯业务的人都会更偏向这类实际的问题吧（再次强调没人比我更懂业务，因为我多年来做的工作其实一直是混合产品经理+设计师+开发的职位）。
面完还和面试官吐槽了之前面试很喜欢让人背诵理论知识。

# 企业面试

啥？还没企业面试我呢，而且是全英文面试，平台也不能提供帮助，只能靠自己了，如果遇到口音难懂的，建议开Zoom的会议实时字幕，或者Window10/11自带的实时字幕，方便帮助理解，如果听不懂也看不懂英文的话，那只能是随缘发挥了。

# 俺很强，俺也要报名！

如果大家也想着试试这种远程工作的话，可以尝试brix等雇佣平台：
[brix工程师入驻](https://engineer.joinbrix.com.cn/)
[Boss直聘企业主页（背后运营的企业1）](https://www.zhipin.com/gongsi/49883974d93e0f881XZ80961E1o~.html)
南京白利度科技和合肥粒度智能科技都是他们注册的企业，要注意他们招聘的其实都是平台入驻人员，而不是内部工作人员。
