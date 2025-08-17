const fs = require('fs');
const marked = require('marked');
const { Pool } = require('pg');

// 配置PostgreSQL连接池
const pool = new Pool({
    user: 'admin',
    host: '192.168.1.194',
    database: 'note',
    password: '123123',
    port: 5432,
});

// 解析Markdown文件
const markdownContent = fs.readFileSync('watch-video.md', 'utf8');
const ast = marked.parse(markdownContent);

// 定义分类和标签
const categories = [];
const tags = [];

// 解析并插入分类
function insertCategory(name) {
    return pool.query('INSERT INTO "Video_Category" (name) VALUES ($1) RETURNING id', [name])
        .then(res => res.rows[0].id);
}

// 解析并插入标签
function insertTag(tag) {
    return pool.query('INSERT INTO "Video_Tag" (tag) VALUES ($1) RETURNING id', [tag])
        .then(res => res.rows[0].id);
}

// 向数据库插入视频数据
function insertVideo(videoData) {
    return pool.query('INSERT INTO "Video" (series, title, link, season, episode, category_id, tags, subtitle, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)', [
        videoData.series,
        videoData.title,
        videoData.link,
        videoData.season,
        videoData.episode,
        videoData.category_id,
        videoData.tags,
        videoData.subtitle,
        videoData.created_at,
    ]);
}

// 解析AST并提取分类和视频信息
function processAST(ast) {
    let currentCategory = null;

    ast.children.forEach(child => {
        if (child.type === 'heading' && child.depth === 2) {
            const categoryName = child.text.replace(/#.*#/, '').trim();
            if (categoryName) {
                insertCategory(categoryName).then(id => {
                    currentCategory = id;
                });
            }
        } else if (child.type === 'list') {
            processList(child, currentCategory);
        }
    });
}

// 处理列表项，根据规则提取视频信息
function processList(list, category_id) {
    list.items.forEach(item => {
        const text = item.text.trim();
        const parts = text.split('、');
        
        // 拆分出基础标题和纯数字部分
        let baseTitle = '';
        const seasons = [];
        
        parts.forEach(part => {
            const cleanPart = part.replace(/\[\S+\]/g, '').trim(); // 移除[]
            if (/^\d+$/.test(cleanPart)) { // 纯数字视为季数
                seasons.push(parseInt(cleanPart));
            } else if (!baseTitle) { // 首个非数字部分作为基础标题
                baseTitle = cleanPart;
            }
        });

        // 没有明确数字季则默认单季
        if (seasons.length === 0) seasons.push(1);

        // 处理多季情况
        seasons.forEach(season => {
            const [title, subtitle] = baseTitle.split(':').map(s => s.trim());
            const linkMatch = title.match(/\((\S+)\)/);
            const link = linkMatch ? linkMatch[1] : '';
            const finalTitle = title.replace(/[\[\]\(\)]/g, '').trim();

            // 生成标签
            const generateTags = (title) => {
                return title.split(/[:\s]+/).filter(word => word.length > 2);
            };

            // 插入标签和视频
            Promise.all(
                generateTags(finalTitle).map(tag => 
                    insertTag(tag).catch(() => null) // 忽略重复标签错误
                )
            ).then(tagIds => {
                const validTagIds = tagIds.filter(id => id);
                insertVideo({
                    series: finalTitle,
                    title: `${finalTitle} 第${season}季`,
                    link: link,
                    season: season,
                    episode: 1, // 默认单集
                    category_id: category_id,
                    tags: validTagIds,
                    subtitle: subtitle || '',
                    created_at: new Date(),
                }).catch(err => console.error('Insert error:', err));
            });
        });
    });
}

// 处理AST
processAST(ast);

// 释放连接池资源
pool.end();
