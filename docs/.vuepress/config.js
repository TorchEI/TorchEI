module.exports = {
  plugins: [["vuepress-plugin-code-copy", true]],
  base: "/TorchEI/",
  locales: {
    "/": {
      lang: "en-US", // 将会被设置为 <html> 的 lang 属性
      title: "TorchEI",
      description:
        "TorchEI, a high-speed toolbox for DNN Reliability's Research and Development",
    },
    "/zh/": {
      lang: "zh-CN",
      title: "TorchEI",
      description: "TorchEI, 一个围绕DNN Reliability的研究和开发的高速工具包",
    },
  },
  themeConfig: {
    sidebarDepth: 3,
    logo: "https://github.com/TorchEI/TorchEI/raw/main/assets/torchei.png",
    smoothScroll: true,
    lastUpdated: "Last Updated",
    repo: "TorchEI/TorchEI",
    sidebar: [
      {
        title: "TorchEI",
        path: "/",
        collapsable: false,
        children: [
          { title: "Preface", path: "/" },
          { title: "Api Reference", path: "/api" },
          //{ title: 'Fault Model', path: '/Fault-Model' },
          //{ title: 'Error Injection', path: '/Error-Injection' },
          //{ title: 'Model Protection', path: '/Model-Protection' },
          //{ title: 'Implemented Algorithms', path: '/Implemented-Algorithms' },

        ]
      },
    ]
  },
  head: [
    ['link', { rel: "apple-touch-icon", sizes: "180x180", href: "/assets/apple-icon-180x180.png" }],
    ['link', { rel: "icon", type: "image/png", sizes: "32x32", href: "/assets/favicon-32x32.png" }],
    ['link', { rel: "icon", type: "image/png", sizes: "16x16", href: "/assets/favicon-16x16.png" }],
    ['link', { rel: "shortcut icon", href: "/assets/favicon.ico" }],
    ['meta', { name: "msapplication-TileColor", content: "#3a0839" }],
    ['meta', { name: "msapplication-config", content: "/assets/browserconfig.xml" }],
    ['meta', { name: "theme-color", content: "#ffffff" }],
  ]
}
