module.exports = {
  base:'/TorchEI/',
  locales: {
    '/': {
      lang: 'en-US', // 将会被设置为 <html> 的 lang 属性
      title: 'TorchEI',
      description: 'TorchEI, a high-speed toolbox around DNN Reliability\'s Research and Development'
    },
    '/zh/': {
      lang: 'zh-CN',
      title: 'TorchEI',
      description: 'TorchEI, 一个围绕DNN Reliability的研究和开发的高速工具包'
    },
  },
  themeConfig: {
    logo: 'https://github.com/TorchEI/TorchEI/raw/main/assets/torchei.png',
    nav: [
      { text: 'HomePage', link: '/' },
      { text: 'Github', link: 'https://github.com/TorchEI/TorchEI' },
    ],
    sidebar: [
      {
        title: 'TorchEI',
        path: '/',
        collapsable: false,
        children: [
          { title: 'Preface', path: '/' },
          { title: 'Fault Model', path: '/Fault-Model' },
          { title: 'Error Injection', path: '/Error-Injection' },
          { title: 'Model Protection', path: '/Model-Protection' },
          { title: 'Implemented Algorithms', path: '/Implemented-Algorithms' },
        ]
      },
    ]
  }
}