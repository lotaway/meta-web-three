package com.metawebthree.media;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.test.context.junit4.SpringRunner;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.github.yulichang.wrapper.MPJLambdaWrapper;
import com.metawebthree.media.DO.ArtWorkDO;
import com.metawebthree.media.DO.PeopleDO;

@RunWith(SpringRunner.class)
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.DEFINED_PORT)
@ComponentScan(basePackages = {
        "com.metawebthree.media",
        "com.metawebthree.common"
})
@EnableAutoConfiguration
public class MediaMapperTest {

    @Autowired
    private ArtWorkMapper artWorkMapper;

    @Autowired
    private PeopleMapper peopleMapper;

    @Test
    public void testCompleteArtWork() {
        // UpdateWrapper<PeopleDO> wrapper = new UpdateWrapper<>();
        // wrapper.eq("name", "文舟").set("types", new Short[]{Short.valueOf("1")});
        // int result = peopleMapper.update(wrapper);
        // Assert.assertEquals(1, result);
        MPJLambdaWrapper<PeopleDO> wrapper = new MPJLambdaWrapper<>();
        wrapper.select(PeopleDO::getId).eq(PeopleDO::getName, "文舟");
        PeopleDO peopleDO = peopleMapper.selectOne(wrapper);
        UpdateWrapper<ArtWorkDO> updateWrapper = new UpdateWrapper<>();
        updateWrapper.eq("series", "骑士的沙丘").set("director", peopleDO.getId());
        int result2 = artWorkMapper.update(updateWrapper);
        Assert.assertTrue(result2 > 0);
        Assert.assertNotNull(artWorkMapper.selectById(11));
    }
}
