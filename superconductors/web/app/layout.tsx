import './globals.css'
import { Inter } from 'next/font/google'
import Link from 'next/link'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'Superconductor Research Lab',
  description: 'Personal superconductor discovery platform',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gray-50">
          <nav className="bg-white shadow-sm border-b">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between h-16">
                <div className="flex items-center">
                  <Link href="/" className="text-xl font-bold text-superconductor-600">
                    🧪 Superconductor Lab
                  </Link>
                </div>
                <div className="flex items-center space-x-4">
                  <Link 
                    href="/experiments" 
                    className="text-gray-700 hover:text-superconductor-600 px-3 py-2 rounded-md text-sm font-medium"
                  >
                    Experiments
                  </Link>
                  <Link 
                    href="/discoveries" 
                    className="text-gray-700 hover:text-superconductor-600 px-3 py-2 rounded-md text-sm font-medium"
                  >
                    Discoveries
                  </Link>
                  <Link 
                    href="/analysis" 
                    className="text-gray-700 hover:text-superconductor-600 px-3 py-2 rounded-md text-sm font-medium"
                  >
                    Analysis
                  </Link>
                </div>
              </div>
            </div>
          </nav>
          
          <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
            {children}
          </main>
        </div>
      </body>
    </html>
  )
}