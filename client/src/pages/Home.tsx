import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Camera, Users, Brain, ArrowRight, Shield, Clock, CheckCircle, Eye, Cpu, Zap, Database } from 'lucide-react';
import Footer from '../components/Footer';

// Header Component
const Header = () => {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-black/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <Eye className="w-8 h-8 text-blue-500 mr-4" />
            <span className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-600 text-transparent bg-clip-text">512D</span>
          </div>
          <nav className="hidden md:flex space-x-8">
            {["Home", "Features", "Pricing", "Contact"].map((item) => (
              <a key={item} href={`#${item.toLowerCase()}`} className="text-gray-300 hover:text-white transition-colors">
                {item}
              </a>
            ))}
          </nav>
          <div>
            <Link to="/login" className="px-4 py-2 rounded-md bg-blue-600 text-white hover:bg-blue-700 transition-colors">
              Login
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
};

// Feature Card Component
const FeatureCard = ({ icon: Icon, title, description }) => {
  return (
    <div className="p-6 bg-white/5 backdrop-blur-md rounded-xl border border-white/10 hover:bg-white/10 transition-all duration-300 hover:scale-105 hover:shadow-lg hover:shadow-blue-500/20">
      <div className="w-16 h-16 bg-blue-500/20 text-blue-400 rounded-full flex items-center justify-center mb-6">
        <Icon className="w-8 h-8" />
      </div>
      <h3 className="text-xl font-bold mb-3 text-white">{title}</h3>
      <p className="text-gray-300">{description}</p>
    </div>
  );
};

// Enhanced Glowing Button Component
const GlowingButton = ({ to, children, primary = false }) => {
  return (
    <Link
      to={to}
      className={`px-8 py-4 rounded-lg font-bold text-lg relative overflow-hidden group ${
        primary 
          ? "bg-gradient-to-r from-blue-600 to-purple-600 text-white" 
          : "bg-white/10 backdrop-blur-md text-white border border-white/20"
      }`}
    >
      <span className="relative z-10 flex items-center">
        {children}
        <ArrowRight className="w-5 h-5 ml-2 transform group-hover:translate-x-1 transition-transform" />
      </span>
      <span className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-500 opacity-0 group-hover:opacity-100 transition-opacity duration-300 blur-xl" />
      <span className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-500 opacity-0 group-hover:opacity-20 transition-opacity duration-300" />
    </Link>
  );
};

// Main Component
export default function Home() {
  const [scrollY, setScrollY] = useState(0);
  const [isLoaded, setIsLoaded] = useState(false);

  // Track scroll position for parallax effects
  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY);
    };
    window.addEventListener('scroll', handleScroll);
    
    // Set loaded state after a short delay to ensure smooth animations
    const loadTimer = setTimeout(() => setIsLoaded(true), 500);
    
    return () => {
      window.removeEventListener('scroll', handleScroll);
      clearTimeout(loadTimer);
    };
  }, []);

  return (
    <div className={`min-h-screen bg-black text-white overflow-hidden transition-opacity duration-1000 ${isLoaded ? 'opacity-100' : 'opacity-0'}`}>
      <Header />  
     
      {/* Hero Section with static background */}
      <section className="h-screen relative flex items-center" id="home">
        <div className="absolute inset-0 z-0">
          <div className="w-full h-full bg-gradient-to-b from-black via-blue-900/30 to-black opacity-80" />
        </div>
        
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid grid-cols-1 lg:grid-cols-2 gap-12 items-center relative z-10">
          <div className="transform transition-all duration-1000 translate-y-0" style={{ transform: `translateY(${-scrollY * 0.1}px)` }}>
            <h1 className="text-6xl font-bold leading-tight mb-6">
              <span className="block mb-2">Revolutionizing</span>
              <span className="bg-gradient-to-r from-blue-400 to-purple-600 text-transparent bg-clip-text">Face Recognition</span>
            </h1>
            <p className="text-xl text-gray-300 mb-10 max-w-lg">
              Experience the future of security and access control with our advanced AI-powered 
              face recognition system delivering unparalleled accuracy and speed.
            </p>
            <div className="flex gap-6">
              <GlowingButton to="/live" primary>Try Live Demo</GlowingButton>
              <GlowingButton to="/attendance">View Attendance</GlowingButton>
            </div>
          </div>
          <div className="h-96 transform transition-all duration-1000" style={{ transform: `translateY(${-scrollY * 0.05}px) scale(${1.2 + scrollY * 0.0005})` }}>
            {/* Placeholder for visual content */}
            <div className="w-full h-full bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl flex items-center justify-center ml-10">
              <Eye className="w-32 h-32 text-white opacity-50" />
            </div>
          </div>
        </div>
        
        <div className="absolute bottom-10 left-0 right-0 flex justify-center">
          <a href="#features" className="animate-bounce bg-white/10 backdrop-blur-md p-2 rounded-full">
            <ArrowRight className="w-6 h-6 text-white transform rotate-90" />
          </a>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 bg-gradient-to-b from-black via-blue-900/10 to-black" id="features">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-20">
            <h2 className="text-4xl font-bold mb-4">Cutting-Edge Features</h2>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              Our system combines advanced AI algorithms with state-of-the-art hardware 
              to deliver a seamless and secure experience.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <FeatureCard
              icon={Camera}
              title="Multi-Camera Integration"
              description="Seamlessly integrate and manage multiple cameras across your facility for comprehensive coverage and monitoring."
            />
            <FeatureCard
              icon={Shield}
              title="Enterprise-Grade Security"
              description="Military-grade encryption and advanced threat detection to ensure your data remains protected at all times."
            />
            <FeatureCard
              icon={Brain}
              title="Advanced AI Processing"
              description="Our proprietary neural networks deliver 99.9% accuracy in face recognition even in challenging lighting conditions."
            />
            <FeatureCard
              icon={Zap}
              title="Real-Time Processing"
              description="Sub-second response times ensure immediate access control decisions without compromising security."
            />
            <FeatureCard
              icon={Database}
              title="Comprehensive Analytics"
              description="Detailed reports and insights on attendance, access patterns, and security events."
            />
            <FeatureCard
              icon={Cpu}
              title="Edge Computing"
              description="Process data locally on-device to reduce latency and enhance privacy protection."
            />
          </div>
        </div>
      </section>

      {/* Stats Section with animated counters */}
      <section className="py-24 bg-blue-900/10 backdrop-blur-lg relative overflow-hidden" id="stats">
        {/* Animated background dots */}
        <div className="absolute inset-0 overflow-hidden">
          {Array.from({ length: 50 }).map((_, i) => (
            <div 
              key={i}
              className="absolute rounded-full bg-blue-500/20"
              style={{
                width: `${Math.random() * 20 + 5}px`,
                height: `${Math.random() * 20 + 5}px`,
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animation: `pulse ${Math.random() * 3 + 2}s infinite alternate ${Math.random() * 2}s`
              }}
            />
          ))}
        </div>
        
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <h2 className="text-4xl font-bold text-center mb-16">The 512D Advantage</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              { number: "99.9%", label: "Recognition Accuracy", color: "from-blue-400 to-blue-600" },
              { number: "0.3s", label: "Response Time", color: "from-purple-400 to-purple-600" },
              { number: "10,000+", label: "Daily Identifications", color: "from-teal-400 to-teal-600" },
              { number: "24/7", label: "Support Available", color: "from-red-400 to-red-600" }
            ].map((stat, index) => (
              <div 
                key={index} 
                className="p-8 bg-white/5 backdrop-blur-md rounded-xl border border-white/10 transform hover:scale-105 transition-all duration-500 hover:shadow-lg hover:shadow-blue-500/20"
              >
                <div className={`text-5xl font-bold mb-4 bg-gradient-to-r ${stat.color} text-transparent bg-clip-text`}>
                  {stat.number}
                </div>
                <div className="text-xl text-gray-300">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonial Section with improved styling */}
      <section className="py-24 bg-gradient-to-b from-black via-blue-900/10 to-black">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-4xl font-bold text-center mb-16">What Our Clients Say</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                quote: "The accuracy and speed of 512D's face recognition system has transformed our security operations.",
                author: "Sarah Johnson",
                role: "Security Director, TechCorp"
              },
              {
                quote: "Implementation was seamless and the support team was there every step of the way. Highly recommended.",
                author: "Michael Chen",
                role: "CTO, FutureTech Solutions"
              },
              {
                quote: "The attendance tracking feature has saved us countless hours and improved employee accountability.",
                author: "Emily Rodriguez",
                role: "HR Manager, Global Enterprises"
              }
            ].map((testimonial, index) => (
              <div 
                key={index} 
                className="p-8 bg-white/5 backdrop-blur-md rounded-xl border border-white/10 hover:bg-white/10 transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/20"
              >
                <div className="text-blue-400 text-6xl font-serif mb-4">"</div>
                <p className="text-lg text-gray-300 mb-6">{testimonial.quote}</p>
                <div>
                  <div className="font-bold text-white">{testimonial.author}</div>
                  <div className="text-blue-400">{testimonial.role}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section className="py-24 bg-gradient-to-b from-black via-blue-900/10 to-black" id="pricing">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-20">
            <h2 className="text-4xl font-bold mb-4">Flexible Pricing Plans</h2>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              Choose the plan that works best for your organization's needs and scale up as you grow.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                name: "Starter",
                price: "$499",
                period: "per month",
                description: "Perfect for small businesses and startups",
                features: [
                  "Up to 5 cameras",
                  "Basic face recognition",
                  "Cloud storage (30 days)",
                  "Email support"
                ],
                buttonText: "Get Started",
                highlight: false
              },
              {
                name: "Professional",
                price: "$999",
                period: "per month",
                description: "Ideal for growing businesses and medium-sized companies",
                features: [
                  "Up to 20 cameras",
                  "Advanced face recognition",
                  "Cloud storage (90 days)",
                  "Priority support",
                  "API access"
                ],
                buttonText: "Choose Plan",
                highlight: true
              },
              {
                name: "Enterprise",
                price: "Custom",
                period: "pricing",
                description: "Tailored solutions for large organizations",
                features: [
                  "Unlimited cameras",
                  "Premium face recognition",
                  "Extended cloud storage",
                  "24/7 dedicated support",
                  "Custom integrations",
                  "On-premise option"
                ],
                buttonText: "Contact Sales",
                highlight: false
              }
            ].map((plan, index) => (
              <div 
                key={index} 
                className={`p-8 rounded-xl border transition-all duration-300 hover:scale-105 ${
                  plan.highlight 
                    ? 'bg-gradient-to-b from-blue-600 to-purple-600 border-white/20 shadow-xl shadow-blue-500/30' 
                    : 'bg-white/5 backdrop-blur-md border-white/10 hover:bg-white/10'
                }`}
              >
                <div className="text-center mb-8">
                  <h3 className="text-2xl font-bold mb-2">{plan.name}</h3>
                  <div className="flex justify-center items-baseline mb-2">
                    <span className="text-4xl font-bold">{plan.price}</span>
                    <span className="text-gray-400 ml-1">/{plan.period}</span>
                  </div>
                  <p className="text-gray-300">{plan.description}</p>
                </div>
                
                <ul className="space-y-3 mb-8">
                  {plan.features.map((feature, i) => (
                    <li key={i} className="flex items-start">
                      <CheckCircle className={`w-5 h-5 mr-3 flex-shrink-0 ${plan.highlight ? 'text-white' : 'text-blue-500'}`} />
                      <span>{feature}</span>
                    </li>
                  ))}
                </ul>
                
                <div className="text-center">
                  <Link
                    to={index === 2 ? "/contact" : "/signup"}
                    className={`inline-block px-6 py-3 rounded-lg font-bold transition-all ${
                      plan.highlight
                        ? 'bg-white text-blue-600 hover:bg-gray-100'
                        : 'bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:opacity-90'
                    }`}
                  >
                    {plan.buttonText}
                  </Link>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-gradient-to-r from-blue-900 to-purple-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-4xl font-bold mb-8">Ready to Experience the Future?</h2>
          <p className="text-xl text-gray-300 mb-12 max-w-2xl mx-auto">
            Join thousands of satisfied clients who have upgraded their security and access control systems with 512D's advanced face recognition technology.
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-6">
            <GlowingButton to="/demo" primary>Schedule a Demo</GlowingButton>
            <GlowingButton to="/contact">Contact Sales</GlowingButton>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="py-24 bg-black" id="faq">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Frequently Asked Questions</h2>
            <p className="text-xl text-gray-300">Find answers to common questions about our technology and services</p>
          </div>
          
          <div className="space-y-6">
            {[
              {
                question: "How accurate is 512D's face recognition technology?",
                answer: "Our face recognition technology achieves 99.9% accuracy in standard conditions and maintains over 95% accuracy in challenging environments including poor lighting, partial occlusion, and various angles."
              },
              {
                question: "Can 512D integrate with existing security systems?",
                answer: "Yes, 512D is designed with compatibility in mind. We offer seamless integration with most major security systems, access control platforms, and HR management software through our comprehensive API."
              },
              {
                question: "What happens if someone tries to use a photo to trick the system?",
                answer: "512D features advanced liveness detection that can distinguish between a real person and a photo, video, or mask. Our system uses depth sensing, texture analysis, and motion detection to prevent spoofing attempts."
              },
              {
                question: "How does 512D handle data privacy and security?",
                answer: "We take data privacy extremely seriously. All biometric data is encrypted using military-grade encryption both in transit and at rest. Our system is compliant with GDPR, CCPA, and other major privacy regulations worldwide."
              },
              {
                question: "What kind of hardware requirements does 512D have?",
                answer: "Our system is designed to work with a wide range of camera hardware. While high-definition cameras will provide optimal results, our AI algorithms can enhance and process images from standard security cameras as well."
              }
            ].map((item, index) => (
              <div key={index} className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
                <h3 className="text-xl font-bold mb-3">{item.question}</h3>
                <p className="text-gray-300">{item.answer}</p>
              </div>
            ))}
          </div>
          
          <div className="mt-12 text-center">
            <p className="text-gray-300 mb-6">Don't see your question here?</p>
            <Link to="/contact" className="text-blue-400 hover:text-blue-300 font-semibold flex items-center justify-center">
              Contact our support team <ArrowRight className="ml-2 w-4 h-4" />
            </Link>
          </div>
        </div>
      </section>
      
      {/* Contact Section */}
      <section className="py-24 bg-gradient-to-b from-black via-blue-900/10 to-black" id="contact">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
            <div>
              <h2 className="text-4xl font-bold mb-6">Get in Touch</h2>
              <p className="text-xl text-gray-300 mb-12">
                Have questions about our technology or services? Our team of experts is ready to help you.
              </p>
              
              <div className="space-y-8">
                {[
                  { icon: Brain, title: "Sales Inquiries", description: "Interested in our solutions? Contact our sales team for pricing and demos.", contact: "sales@512d.ai" },
                  { icon: Shield, title: "Technical Support", description: "Need help with your implementation? Our support engineers are available 24/7.", contact: "support@512d.ai" },
                  { icon: Users, title: "Partnerships", description: "Looking to partner with us? Let's explore opportunities together.", contact: "partners@512d.ai" }
                ].map((item, index) => (
                  <div key={index} className="flex items-start">
                    <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center mr-4 flex-shrink-0">
                      <item.icon className="w-6 h-6 text-blue-400" />
                    </div>
                    <div>
                      <h3 className="text-xl font-bold mb-1">{item.title}</h3>
                      <p className="text-gray-300 mb-2">{item.description}</p>
                      <a href={`mailto:${item.contact}`} className="text-blue-400 hover:text-blue-300">{item.contact}</a>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="bg-white/5 backdrop-blur-sm p-8 rounded-xl border border-white/10">
              <h3 className="text-2xl font-bold mb-6">Send us a Message</h3>
              <form className="space-y-6">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                  <div>
                    <label htmlFor="name" className="block text-sm font-medium text-gray-300 mb-2">Name</label>
                    <input
                      type="text"
                      id="name"
                      className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-white"
                      placeholder="Your name"
                    />
                  </div>
                  <div>
                    <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-2">Email</label>
                    <input
                      type="email"
                      id="email"
                      className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-white"
                      placeholder="your@email.com"
                    />
                  </div>
                </div>
                
                <div>
                  <label htmlFor="company" className="block text-sm font-medium text-gray-300 mb-2">Company</label>
                  <input
                    type="text"
                    id="company"
                    className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-white"
                    placeholder="Your company"
                  />
                </div>
                
                <div>
                  <label htmlFor="message" className="block text-sm font-medium text-gray-300 mb-2">Message</label>
                  <textarea
                    id="message"
                    rows={5}
                    className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-white"
                    placeholder="How can we help you?"
                  ></textarea>
                </div>
                
                <div>
                  <button
                    type="submit"
                    className="w-full py-3 px-6 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold rounded-lg hover:opacity-90 transition-all"
                  >
                    Send Message
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      </section>
      
      {/* Loading Overlay - Hidden after initial loading */}
      <div className={`fixed inset-0 bg-black z-[100] flex items-center justify-center transition-opacity duration-1000 ${isLoaded ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}>
        <div className="flex flex-col items-center">
          <Eye className="w-16 h-16 text-blue-500 animate-pulse" />
          <div className="mt-4 text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-600 text-transparent bg-clip-text">
            512D
          </div>
          <div className="mt-2 text-gray-400">Loading experience...</div>
          <div className="mt-8 w-48 h-1 bg-white/10 rounded overflow-hidden">
            <div className="h-full bg-gradient-to-r from-blue-500 to-purple-500 animate-[loading_2s_ease-in-out_infinite]"></div>
          </div>
        </div>
      </div>

      {/* Add global style for the loading animation */}
      <style jsx>{`
        @keyframes loading {
          0% { width: 0; transform: translateX(-100%); }
          50% { width: 100%; transform: translateX(0); }
          100% { width: 0; transform: translateX(100%); }
        }
        @keyframes pulse {
          0% { opacity: 0.2; transform: scale(0.8); }
          100% { opacity: 0.6; transform: scale(1.2); }
        }
      `}</style>

      <Footer />
    </div>
  );
}