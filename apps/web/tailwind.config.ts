import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",        // App Router
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}", // Local components
    // legacy fallbacks
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",      // (if any legacy pages remain)
  ],
  theme: {
    extend: {
      fontFamily: {
        poppins: ['Poppins', 'sans-serif'],
      
      },
      fontWeight: {
        light: '300',
        normal: '400',
        medium: '500',
        semibold: '600',
        bold: '700',
      },
      borderRadius: {
        '4xl': '5rem', 
      },
      colors: {
        'custom-green': '#287571',
        'custom-green-end': '#88D3D0',
      },
      backgroundImage: {
        'gradient-to-b': 'linear-gradient(to bottom, #287571, #88D3D0)',
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic":
          "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
      },
      textColor: {
        'gradient-green-white': 'transparent',
      },
      backgroundClip: {
        text: 'text',
      },
    },
  },
  plugins: [],
};
export default config;