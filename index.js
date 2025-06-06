const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');

const client = new Client({
    authStrategy: new LocalAuth()
});

client.on('qr', (qr) => {
    console.log('📱 Scan this QR code with your WhatsApp:');
    qrcode.generate(qr, { small: true });
});

client.on('ready', () => {
    console.log('✅ WhatsApp bot is ready!');
});

client.on('message', async (message) => {
    const text = message.body.toLowerCase();
    console.log(`📥 ${message.from}: ${text}`);

    if (text.includes('hola')) {
        await message.reply('¡Hola! 👋 Soy tu asistente Espaluz. ¿Quieres aprender español hoy?');
    } else {
        await message.reply('Escríbeme "hola" para comenzar. 📚');
    }
});

client.initialize();
