<template>
    <section>
        <h2>{{name}}</h2>
        <ul class="list">
            <li class="item">
                <nuxt-link to="/test">测试</nuxt-link>
            </li>
        </ul>
    </section>
</template>

<script>
    export default {
        asyncData(context) {
            // called every time before loading the component
            return {
                name: "这是页面的名称"
            }
        },
        fetch() {
            // The fetch method is used to fill the store before rendering the page
        },
        head() {
            // Set Meta Tags for this Page
        },
        // and more functionality to discover
    };
</script>