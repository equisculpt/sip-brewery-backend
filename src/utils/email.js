const SibApiV3Sdk = require('@sendinblue/client');

const apiKey = process.env.BREVO_API_KEY || process.env.SENDINBLUE_API_KEY;
const defaultFrom = { email: 'no-reply@sipbrewery.com', name: 'SIP Brewery' };

const client = new SibApiV3Sdk.TransactionalEmailsApi();
if (apiKey) client.setApiKey(SibApiV3Sdk.TransactionalEmailsApiApiKeys.apiKey, apiKey);

async function sendOtpEmail({ to, otp }) {
  if (!apiKey) {
    console.error('Brevo API key not set. Cannot send email.');
    return;
  }
  try {
    await client.sendTransacEmail({
      sender: defaultFrom,
      to: [{ email: to }],
      subject: 'Your SIP Brewery OTP',
      htmlContent: `<p>Your OTP is <b>${otp}</b>. It is valid for 5 minutes.</p>`
    });
    console.log(`OTP email sent to ${to}`);
  } catch (err) {
    console.error('Failed to send OTP email:', err.message);
  }
}

module.exports = { sendOtpEmail }; 