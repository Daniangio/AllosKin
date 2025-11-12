/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        // Add 'Inter' font to match the App.js style
        sans: ['Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
}