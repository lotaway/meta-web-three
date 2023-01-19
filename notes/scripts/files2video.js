//  todo @require("./files2video.pdf") 根据文档完成文件输出成视频的需求
const fs = require("fs")
const path = require("path")
const generaPath = path.join(__dirname, "./General")
const platformPath = path.join(__dirname, "./Platform")
const widgetsPath = path.join(__dirname, "./Widgets")
const destinationFolder = path.join(__dirname, "./build")

async function getIncludeFiles(path) {
    return await new Promise((resolve, reject) => {
        let filenames = []
        const items = fs.readdirSync(path, {
            withFileTypes: true
        })
        if (items.length === 0) resolve([])
        items.forEach((item, index) => {
            items.isFile() && filenames.push(items.name)
            items.splice(index, 1)
            items.length === 0 && resolve(filenames)
        })
    })
}

function filename2path(filenames, prevFix) {
    return filenames.map(filename => path.join(prevFix, filename))
}

getIncludeFiles(generaPath).then(filenames => {
    const filePaths = filename2path(filenames, generaPath)
})

getIncludeFiles(platformPath).then(filenames => {
    const filePaths = filename2path(filenames, generaPath)
})

getIncludeFiles(widgetsPath).then(filenames => {
    const filePaths = filename2path(filenames, generaPath)
})